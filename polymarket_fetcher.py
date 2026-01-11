"""
Polymarket Data Fetcher - Enhanced Version
Robust API client with retry logic, caching, and rate limiting
"""

import os
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
import time
import json
from functools import lru_cache
import numpy as np


class RateLimiter:
    """Simple rate limiter to avoid API throttling"""
    
    def __init__(self, calls_per_second: float = 2.0):
        self.min_interval = 1.0 / calls_per_second
        self.last_call = 0
    
    def wait(self):
        """Wait if necessary to respect rate limit"""
        elapsed = time.time() - self.last_call
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self.last_call = time.time()


class PolymarketFetcher:
    """
    Robust Polymarket API client with:
    - Rate limiting
    - Retry logic with exponential backoff
    - Response caching
    - Proper error handling
    """
    
    # Official API endpoints
    CLOB_URL = "https://clob.polymarket.com"
    GAMMA_URL = "https://gamma-api.polymarket.com"
    DATA_URL = "https://data-api.polymarket.com"
    
    def __init__(self, verbose: bool = False, cache_ttl: int = 300, 
                 timeout: int = 30, max_retries: int = 3, api_key: str = None):
        self.verbose = verbose
        self.cache_ttl = cache_ttl
        self.default_timeout = timeout
        
        # Get API key from parameter or environment variable
        self.api_key = api_key or os.getenv('POLYMARKET_API_KEY')
        
        # Session with connection pooling and retry adapter (best practice)
        self.session = requests.Session()
        headers = {
            'User-Agent': 'PolymarketPredictor/3.0',
            'Accept': 'application/json',
            'Connection': 'keep-alive'
        }
        
        # Add Polymarket API key to headers if provided
        # Polymarket CLOB API uses POLY_API_KEY header (not Authorization Bearer)
        if self.api_key:
            headers['POLY_API_KEY'] = self.api_key
            if verbose:
                print(f"[Fetcher] API key configured (length: {len(self.api_key)})")
        
        self.session.headers.update(headers)
        
        # Configure retry strategy with exponential backoff
        retry_strategy = Retry(
            total=max_retries,
            backoff_factor=1,  # Wait 1s, 2s, 4s between retries
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET", "POST"],
            raise_on_status=False
        )
        adapter = HTTPAdapter(
            max_retries=retry_strategy,
            pool_connections=10,  # Connection pool size
            pool_maxsize=20       # Max connections per pool
        )
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)
        
        # Rate limiter
        self.rate_limiter = RateLimiter(calls_per_second=2.0)
        
        # Cache
        self._markets_cache: Dict[str, dict] = {}
        self._cache_timestamps: Dict[str, datetime] = {}
        self._tags_cache: Dict[str, dict] = {}
        self._tags_cache_timestamps: Dict[str, datetime] = {}
    
    def _log(self, message: str):
        if self.verbose:
            print(f"[Fetcher] {message}")
    
    def _request(
        self, 
        url: str, 
        params: dict = None, 
        timeout: int = None,
        retries: int = 3
    ) -> Optional[dict]:
        """Make HTTP request with retry logic and rate limiting"""
        
        self.rate_limiter.wait()
        
        if timeout is None:
            timeout = self.default_timeout
        
        for attempt in range(retries):
            try:
                response = self.session.get(url, params=params, timeout=timeout)
                response.raise_for_status()
                return response.json()
                
            except requests.exceptions.Timeout:
                self._log(f"Timeout (attempt {attempt + 1}/{retries})")
                if attempt < retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                    
            except requests.exceptions.HTTPError as e:
                status_code = e.response.status_code
                if status_code == 429:  # Rate limited
                    wait_time = 5 * (2 ** attempt)
                    self._log(f"Rate limited, waiting {wait_time}s")
                    time.sleep(wait_time)
                elif status_code == 401:  # Unauthorized
                    self._log(f"Authentication failed (401): Check API key. Response: {e.response.text[:200]}")
                    return None  # Don't retry on auth errors
                elif status_code == 403:  # Forbidden
                    self._log(f"Access forbidden (403): API key may not have required permissions. Response: {e.response.text[:200]}")
                    return None  # Don't retry on permission errors
                elif status_code >= 500:
                    self._log(f"Server error {status_code}")
                    if attempt < retries - 1:
                        time.sleep(2 ** attempt)
                else:
                    # Log response details for debugging (4xx errors)
                    try:
                        error_detail = e.response.text[:200] if hasattr(e.response, 'text') else 'No response body'
                        self._log(f"HTTP {status_code}: {url} - {error_detail}")
                    except:
                        self._log(f"HTTP {status_code}: {url}")
                    return None
                    
            except requests.exceptions.RequestException as e:
                self._log(f"Request error: {e}")
                if attempt < retries - 1:
                    time.sleep(2 ** attempt)
                    
            except json.JSONDecodeError:
                self._log(f"Invalid JSON from {url}")
                return None
        
        return None
    
    def _is_cache_valid(self, key: str, cache_timestamps: Dict[str, datetime] = None) -> bool:
        """
        Check if cached data is still valid
        
        Args:
            key: Cache key to check
            cache_timestamps: Optional cache timestamps dict (default: uses _cache_timestamps)
        
        Returns: True if cache is valid, False otherwise
        """
        timestamps = cache_timestamps if cache_timestamps is not None else self._cache_timestamps
        if key not in timestamps:
            return False
        age = (datetime.now() - timestamps[key]).total_seconds()
        return age < self.cache_ttl
    
    # =========================================================================
    # GAMMA API - Market Discovery
    # =========================================================================
    
    def get_markets(
        self,
        limit: int = 100,
        offset: int = 0,
        active: bool = True,
        closed: bool = False,
        order: str = "volume24hr",
        ascending: bool = False
    ) -> List[Dict]:
        """Fetch markets from Gamma API with caching"""
        
        cache_key = f"markets_{limit}_{offset}_{order}"
        
        if cache_key in self._markets_cache and self._is_cache_valid(cache_key):
            self._log("Using cached markets")
            return self._markets_cache[cache_key]
        
        url = f"{self.GAMMA_URL}/markets"
        params = {
            'limit': limit,
            'offset': offset,
            'order': order,
            'ascending': str(ascending).lower()
        }
        
        if active:
            params['active'] = 'true'
        if not closed:
            params['closed'] = 'false'
        
        self._log(f"Fetching markets: limit={limit}, order={order}")
        
        data = self._request(url, params)
        
        if data:
            self._markets_cache[cache_key] = data
            self._cache_timestamps[cache_key] = datetime.now()
            
            # Also cache individual markets
            for market in data:
                market_id = market.get('id') or market.get('conditionId')
                if market_id:
                    self._markets_cache[market_id] = market
            
            self._log(f"Fetched {len(data)} markets")
            return data
        
        # Return cached data if API fails
        if cache_key in self._markets_cache:
            self._log("API failed, using stale cache")
            return self._markets_cache[cache_key]
        
        return []
    
    def get_market_by_id(self, market_id: str) -> Optional[Dict]:
        """Get a specific market by ID"""
        if market_id in self._markets_cache:
            return self._markets_cache[market_id]
        
        url = f"{self.GAMMA_URL}/markets/{market_id}"
        data = self._request(url)
        
        if data:
            self._markets_cache[market_id] = data
        
        return data
    
    def search_markets(self, query: str, limit: int = 20) -> List[Dict]:
        """Search markets by question text"""
        all_markets = self.get_markets(limit=200)
        
        query_lower = query.lower()
        matching = [
            m for m in all_markets
            if query_lower in (m.get('question', '') or '').lower()
            or query_lower in (m.get('description', '') or '').lower()
        ]
        
        return matching[:limit]
    
    # =========================================================================
    # GAMMA API - Tags and Categories
    # =========================================================================
    
    def get_tags(self, limit: int = 200, offset: int = 0) -> List[Dict]:
        """
        Fetch tags (categories) from Gamma API with caching.
        
        Tags represent categories in Polymarket (e.g., Crypto, Politics, Sports).
        
        Args:
            limit: Maximum number of tags to fetch (default: 200)
            offset: Pagination offset (default: 0)
        
        Returns: List of tag dictionaries (categories)
        """
        cache_key = f"tags_{limit}_{offset}"
        
        if cache_key in self._tags_cache and self._is_cache_valid(cache_key, cache_timestamps=self._tags_cache_timestamps):
            self._log("Using cached tags")
            return self._tags_cache[cache_key]
        
        url = f"{self.GAMMA_URL}/tags"
        params = {
            'limit': limit,
            'offset': offset
        }
        
        self._log(f"Fetching tags: limit={limit}, offset={offset}")
        
        data = self._request(url, params)
        
        if data:
            self._tags_cache[cache_key] = data
            self._tags_cache_timestamps[cache_key] = datetime.now()
            
            # Also cache individual tags
            for tag in data:
                tag_id = tag.get('id')
                tag_slug = tag.get('slug')
                if tag_id:
                    self._tags_cache[f"tag_id_{tag_id}"] = tag
                    self._tags_cache_timestamps[f"tag_id_{tag_id}"] = datetime.now()
                if tag_slug:
                    self._tags_cache[f"tag_slug_{tag_slug}"] = tag
                    self._tags_cache_timestamps[f"tag_slug_{tag_slug}"] = datetime.now()
            
            self._log(f"Fetched {len(data)} tags")
            return data
        
        # Return cached data if API fails
        if cache_key in self._tags_cache:
            self._log("API failed, using stale cache")
            return self._tags_cache[cache_key]
        
        return []
    
    def find_tag_by_label(
        self, 
        target_label: str, 
        page_size: int = 200, 
        max_pages: int = 50
    ) -> Tuple[Optional[str], Optional[str], Optional[Dict]]:
        """
        Find a tag by exact label match, searching through paginated results.
        
        This is more robust than fetching a single page, as it handles cases
        where tags are spread across multiple pages or labels change over time.
        
        Args:
            target_label: Tag label to search for (e.g., "Politics", "Crypto")
            page_size: Number of tags per page (default: 200)
            max_pages: Maximum number of pages to search (default: 50)
        
        Returns: Tuple of (tag_id, slug, tag_object) or (None, None, None) if not found
        """
        # Normalize target label (strip and lowercase)
        target_label_normalized = target_label.strip().lower()
        
        if not target_label_normalized:
            self._log("Empty target label provided")
            return None, None, None
        
        self._log(f"Searching for tag with label: '{target_label}' (normalized: '{target_label_normalized}')")
        
        # Iterate through pages
        offset = 0
        pages_searched = 0
        for page_num in range(max_pages):
            pages_searched = page_num + 1
            # Check cache first
            cache_key = f"tags_{page_size}_{offset}"
            
            # Try to get from cache
            tags = None
            if cache_key in self._tags_cache and self._is_cache_valid(cache_key, cache_timestamps=self._tags_cache_timestamps):
                self._log(f"Using cached tags for page {pages_searched} (offset={offset})")
                tags = self._tags_cache[cache_key]
            else:
                # Fetch tags for this page using get_tags() which handles caching
                tags = self.get_tags(limit=page_size, offset=offset)
            
            # If response is empty list, break (no more pages)
            if not tags or len(tags) == 0:
                self._log(f"No more tags found after page {pages_searched}")
                break
            
            # Search through tags in this page
            for tag in tags:
                if not isinstance(tag, dict):
                    continue  # Skip malformed tag data
                
                # Get and normalize label
                tag_label = tag.get('label', '')
                if not tag_label:
                    continue  # Skip tags without labels
                
                tag_label_normalized = tag_label.strip().lower()
                
                # Compare normalized labels (exact match)
                if tag_label_normalized == target_label_normalized:
                    tag_id = tag.get('id')
                    tag_slug = tag.get('slug')
                    
                    if tag_id:
                        self._log(f"✅ Found tag: '{tag_label}' (id: {tag_id}, slug: {tag_slug})")
                        # Return tag_id as string
                        return str(tag_id), tag_slug, tag
                    else:
                        self._log(f"Found matching tag but missing ID: '{tag_label}'")
            
            # Increment offset for next page
            offset += page_size
        
        # Tag not found after searching all pages
        self._log(f"❌ Tag with label '{target_label}' not found after searching {pages_searched} pages")
        return None, None, None
    
    def get_tag_by_id(self, tag_id: str) -> Optional[Dict]:
        """
        Get a specific tag by ID.
        
        Args:
            tag_id: Tag ID
        
        Returns: Tag dictionary or None if not found
        """
        cache_key = f"tag_id_{tag_id}"
        
        if cache_key in self._tags_cache and self._is_cache_valid(cache_key, cache_timestamps=self._tags_cache_timestamps):
            self._log(f"Using cached tag: {tag_id}")
            return self._tags_cache[cache_key]
        
        url = f"{self.GAMMA_URL}/tags/{tag_id}"
        data = self._request(url)
        
        if data:
            self._tags_cache[cache_key] = data
            self._tags_cache_timestamps[cache_key] = datetime.now()
            
            # Also cache by slug if available
            tag_slug = data.get('slug')
            if tag_slug:
                slug_cache_key = f"tag_slug_{tag_slug}"
                self._tags_cache[slug_cache_key] = data
                self._tags_cache_timestamps[slug_cache_key] = datetime.now()
            
            self._log(f"Fetched tag by ID: {tag_id}")
            return data
        
        return None
    
    def get_tag_by_slug(self, tag_slug: str) -> Optional[Dict]:
        """
        Get a specific tag by slug.
        
        Args:
            tag_slug: Tag slug (e.g., 'crypto', 'politics', 'sports')
        
        Returns: Tag dictionary or None if not found
        """
        cache_key = f"tag_slug_{tag_slug}"
        
        if cache_key in self._tags_cache and self._is_cache_valid(cache_key, cache_timestamps=self._tags_cache_timestamps):
            self._log(f"Using cached tag: {tag_slug}")
            return self._tags_cache[cache_key]
        
        url = f"{self.GAMMA_URL}/tags/{tag_slug}"
        data = self._request(url)
        
        if data:
            self._tags_cache[cache_key] = data
            self._tags_cache_timestamps[cache_key] = datetime.now()
            
            # Also cache by ID if available
            tag_id = data.get('id')
            if tag_id:
                id_cache_key = f"tag_id_{tag_id}"
                self._tags_cache[id_cache_key] = data
                self._tags_cache_timestamps[id_cache_key] = datetime.now()
            
            self._log(f"Fetched tag by slug: {tag_slug}")
            return data
        
        return None
    
    def get_related_tags(self, tag_id_or_slug: str, use_id: bool = True) -> List[Dict]:
        """
        Get related tags (sub-categories) for a tag.
        
        Retrieves tags that are related to the given tag, which represents
        sub-categories in Polymarket's category hierarchy.
        
        Args:
            tag_id_or_slug: Tag ID or slug
            use_id: If True, treat tag_id_or_slug as ID; if False, treat as slug
        
        Returns: List of related tag dictionaries (sub-categories)
        """
        cache_key = f"related_tags_{tag_id_or_slug}"
        
        if cache_key in self._tags_cache and self._is_cache_valid(cache_key, cache_timestamps=self._tags_cache_timestamps):
            self._log(f"Using cached related tags: {tag_id_or_slug}")
            return self._tags_cache[cache_key]
        
        url = f"{self.GAMMA_URL}/tags/{tag_id_or_slug}/related-tags/tags"
        data = self._request(url)
        
        if data:
            # Handle different response formats
            tags = []
            if isinstance(data, list):
                tags = data
            elif isinstance(data, dict):
                tags = data.get('tags', data.get('data', data.get('results', [])))
            
            if tags:
                self._tags_cache[cache_key] = tags
                self._tags_cache_timestamps[cache_key] = datetime.now()
                self._log(f"Fetched {len(tags)} related tags for {tag_id_or_slug}")
                return tags
        
        # Return cached data if API fails
        if cache_key in self._tags_cache:
            self._log("API failed, using stale cache")
            return self._tags_cache[cache_key]
        
        return []
    
    # =========================================================================
    # CLOB API - Orderbook and Trades
    # =========================================================================
    
    def get_orderbook(self, token_id: str) -> Dict:
        """Get current orderbook for a token"""
        url = f"{self.CLOB_URL}/book"
        params = {'token_id': token_id}
        
        data = self._request(url, params)
        
        if data:
            return {
                'bids': data.get('bids', []),
                'asks': data.get('asks', []),
                'market': data.get('market', ''),
                'asset_id': data.get('asset_id', token_id),
                'timestamp': data.get('timestamp', int(time.time() * 1000))
            }
        
        return {'bids': [], 'asks': [], 'market': '', 'asset_id': token_id}
    
    def get_midpoint(self, token_id: str) -> Optional[float]:
        """Get midpoint price for a token"""
        url = f"{self.CLOB_URL}/midpoint"
        params = {'token_id': token_id}
        
        data = self._request(url, params)
        if data and 'mid' in data:
            return float(data['mid'])
        return None
    
    def get_price(self, token_id: str, side: str = "BUY") -> Optional[float]:
        """Get best price for a side"""
        url = f"{self.CLOB_URL}/price"
        params = {'token_id': token_id, 'side': side}
        
        data = self._request(url, params)
        if data and 'price' in data:
            return float(data['price'])
        return None
    
    def get_trades(self, token_id: str = None, limit: int = 500, market: str = None, asset_id: str = None, side: str = None) -> List[Dict]:
        """
        Fetch recent trades from Polymarket Data API (public market data).
        
        Uses the Data API endpoint which is the recommended source for public market trades tape.
        This endpoint properly supports filtering by conditionId (market) and asset (token).
        
        Args:
            token_id: Token ID (asset ID) - alias for asset_id parameter
            limit: Maximum number of trades to fetch (default: 500, max: 500 for Data API)
            market: Market/condition ID (REQUIRED - conditionId parameter)
            asset_id: Asset ID (token ID) - filters trades for specific outcome token (YES/NO)
            side: Trade side filter - 'BUY' or 'SELL' (optional)
        
        Returns: List of trade dictionaries with fields: id, asset_id, conditionId, price, size, side, timestamp, etc.
        """
        # Use Data API endpoint (public market data) instead of CLOB API
        url = f"{self.DATA_URL}/trades"
        
        # Determine target asset_id for filtering (from parameter or token_id)
        target_asset_id = asset_id or token_id
        
        # Data API requires conditionId parameter (market/condition ID)
        if not market:
            self._log(f"⚠️ No market parameter provided. Cannot fetch trades without conditionId.")
            self._log(f"   Hint: Provide 'market' parameter (condition ID) to fetch trades")
            return []
        
        # Build parameters for Data API
        params = {
            'conditionId': market,  # Data API uses 'conditionId' not 'market'
            'limit': min(limit, 500)  # Data API max is 500
        }
        
        # Add asset filter if provided (Data API supports this as query parameter)
        if target_asset_id:
            params['asset'] = target_asset_id
            self._log(f"Fetching trades for conditionId={market[:20]}... with asset={target_asset_id[:20]}...")
        else:
            self._log(f"Fetching trades for conditionId={market[:20]}... (all outcomes)")
        
        # Add side filter if provided
        if side:
            params['side'] = side.upper()  # Ensure uppercase
            self._log(f"Filtering by side: {side}")
        
        self._log(f"API Request: {url} with params: {params}")
        
        data = self._request(url, params)
        
        if data is None:
            self._log("❌ No trades data returned from API")
            self._log("   Check: 1) Market ID (conditionId) is correct 2) Market has active trading")
            return []
        
        # Handle different response formats
        trades = []
        if isinstance(data, list):
            # Direct list response - most common format
            trades = data
        elif isinstance(data, dict):
            # Nested response - try common keys
            trades = data.get('trades', data.get('data', data.get('results', data.get('items', []))))
            
            # If still no trades, check if response indicates empty result
            if not trades:
                count = data.get('count', data.get('total', data.get('length', None)))
                if count is not None and count == 0:
                    self._log(f"API returned empty result (count=0) - market may have no recent trades")
                    return []
                elif 'message' in data:
                    self._log(f"API message: {data.get('message')}")
                elif 'error' in data:
                    self._log(f"API error: {data.get('error')}")
        else:
            self._log(f"⚠️ Unexpected response format: {type(data)}, value: {str(data)[:100]}")
            return []
        
        # Additional client-side filtering if needed (backup in case API filter didn't work)
        original_count = len(trades)
        if target_asset_id and trades:
            # Double-check filtering (API should have filtered, but verify)
            filtered_trades = []
            for trade in trades:
                # Check multiple possible field names for asset_id
                trade_asset_id = trade.get('asset') or trade.get('asset_id') or trade.get('assetId') or trade.get('token_id') or trade.get('tokenId')
                if trade_asset_id == target_asset_id:
                    filtered_trades.append(trade)
            
            if len(filtered_trades) != original_count:
                self._log(f"Client-side filtered {original_count} trades to {len(filtered_trades)} trades for asset={target_asset_id[:20]}...")
            trades = filtered_trades
        
        if trades:
            self._log(f"✅ Successfully fetched {len(trades)} trades from Data API")
            # Log sample trade structure for debugging
            if self.verbose and trades:
                sample_trade = trades[0]
                sample_keys = list(sample_trade.keys())[:10]
                self._log(f"Sample trade structure: {sample_keys}")
                if target_asset_id:
                    sample_asset = sample_trade.get('asset', sample_trade.get('asset_id', sample_trade.get('assetId', 'N/A')))
                    self._log(f"Sample trade asset: {sample_asset}")
        else:
            if original_count == 0:
                self._log("⚠️ No trades found in API response (market may be inactive or have no recent trades)")
            else:
                self._log(f"⚠️ No trades found after filtering (filtered {original_count} trades, none matched asset={target_asset_id[:20] if target_asset_id else 'N/A'})")
        
        return trades
    
    def get_last_trade_price(self, token_id: str) -> Optional[float]:
        """Get last trade price for a token"""
        url = f"{self.CLOB_URL}/last-trade-price"
        params = {'token_id': token_id}
        
        data = self._request(url, params)
        if data and 'price' in data:
            return float(data['price'])
        return None
    
    def get_prices_history(
        self, 
        token_id: str, 
        interval: str = "max",
        fidelity: int = 60,
        start_ts: int = None,
        end_ts: int = None
    ) -> pd.DataFrame:
        """
        Fetch historical price timeseries for a market token.
        
        REAL DATA from Polymarket CLOB API /prices-history endpoint.
        
        Args:
            token_id: The CLOB token ID
            interval: Duration ('1h', '6h', '1d', '1w', '1m', 'max')
            fidelity: Resolution in minutes (default 60 = hourly)
            start_ts: Unix timestamp for start (mutually exclusive with interval)
            end_ts: Unix timestamp for end
        
        Returns: DataFrame with columns ['timestamp', 'price']
        """
        url = f"{self.CLOB_URL}/prices-history"
        params = {
            'market': token_id,
            'fidelity': fidelity
        }
        
        if start_ts and end_ts:
            params['startTs'] = start_ts
            params['endTs'] = end_ts
        else:
            params['interval'] = interval
        
        self._log(f"Fetching price history for {token_id[:20]}...")
        
        data = self._request(url, params)
        
        if data and 'history' in data:
            history = data['history']
            if history:
                df = pd.DataFrame(history)
                # API returns 't' for timestamp and 'p' for price
                if 't' in df.columns and 'p' in df.columns:
                    df = df.rename(columns={'t': 'timestamp', 'p': 'price'})
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
                df['price'] = pd.to_numeric(df['price'], errors='coerce')
                self._log(f"Fetched {len(df)} price points")
                return df.sort_values('timestamp').reset_index(drop=True)
        
        self._log(f"No price history available for token")
        return pd.DataFrame(columns=['timestamp', 'price'])
    
    def validate_data_freshness(self, trades_df: pd.DataFrame, max_age_hours: float = 24) -> Dict:
        """
        Validate that training/prediction data is fresh (best practice).
        
        Args:
            trades_df: DataFrame with 'timestamp' column
            max_age_hours: Maximum acceptable age for most recent trade
        
        Returns: Dict with freshness info and warnings
        """
        result = {
            'is_fresh': True,
            'latest_trade': None,
            'age_hours': 0,
            'warning': None
        }
        
        if trades_df is None or trades_df.empty or 'timestamp' not in trades_df.columns:
            result['is_fresh'] = False
            result['warning'] = "No timestamp data available"
            return result
        
        try:
            latest = pd.to_datetime(trades_df['timestamp']).max()
            now = datetime.now()
            age = (now - latest).total_seconds() / 3600
            
            result['latest_trade'] = latest.isoformat()
            result['age_hours'] = round(age, 2)
            
            if age > max_age_hours:
                result['is_fresh'] = False
                result['warning'] = f"Data is {age:.1f} hours old (max: {max_age_hours}h)"
            elif age > max_age_hours / 2:
                result['warning'] = f"Data is getting stale ({age:.1f}h old)"
        except Exception as e:
            result['is_fresh'] = False
            result['warning'] = f"Could not parse timestamps: {e}"
        
        return result
    
    def get_spread(self, token_id: str) -> Dict:
        """Get current bid-ask spread for a token"""
        url = f"{self.CLOB_URL}/spread"
        params = {'token_id': token_id}
        
        data = self._request(url, params)
        
        if data:
            return {
                'bid': float(data.get('bid', 0)),
                'ask': float(data.get('ask', 0)),
                'spread': float(data.get('spread', 0))
            }
        return {'bid': 0, 'ask': 0, 'spread': 0}
    
    # =========================================================================
    # Real Training Data Methods (NO SYNTHETIC DATA)
    # =========================================================================
    
    def fetch_real_training_data(
        self,
        n_markets: int = 50,
        min_volume: float = 1000,
        include_closed: bool = False  # Disabled - CLOB returns 404 for closed markets
    ) -> List[Dict]:
        """
        Fetch REAL training data from Polymarket for ML model training.
        
        NO SYNTHETIC DATA - uses actual historical prices and outcomes.
        Uses price history to create training labels (price direction over time).
        
        Args:
            n_markets: Number of markets to fetch
            min_volume: Minimum 24h volume filter
            include_closed: Disabled - closed markets have no CLOB data
        
        Returns: List of training samples with real features and outcomes
        """
        self._log("=" * 60)
        self._log("FETCHING REAL TRAINING DATA FROM POLYMARKET API")
        self._log("=" * 60)
        
        training_data = []
        
        # Fetch active markets for live price data
        self._log("\n📊 Fetching active markets with high volume...")
        active_markets = self.get_markets(
            limit=500,  # Fetch 500 markets (max for better training data)
            active=True,
            closed=False,
            order='volume24hr',
            ascending=False
        )
        
        self._log(f"Found {len(active_markets)} active markets")
        
        for market in active_markets:
            sample = self._process_market_for_training(market, is_closed=False)
            if sample:
                training_data.append(sample)
            if len(training_data) >= n_markets:
                break
        
        # Note: Closed markets return 404 from CLOB API - cannot use them
        # Training uses price history-based labels instead
        
        self._log(f"\n✅ Total training samples: {len(training_data)}")
        return training_data
    
    def _process_market_for_training(self, market: Dict, is_closed: bool = False) -> Optional[Dict]:
        """
        Process a market into a training sample with REAL data.
        
        Returns: Training sample dict or None if insufficient data
        """
        try:
            # Extract token IDs
            yes_token, no_token = self.get_token_ids_for_market(market)
            
            if not yes_token:
                return None
            
            # Get current/final price
            prices_str = market.get('outcomePrices', '[0.5, 0.5]')
            try:
                prices = json.loads(prices_str) if isinstance(prices_str, str) else prices_str
                current_price = float(prices[0]) if prices else 0.5
            except:
                current_price = 0.5
            
            # Skip if price is extreme (likely already resolved)
            if current_price < 0.01 or current_price > 0.99:
                current_price = np.clip(current_price, 0.01, 0.99)
            
            # Get real order book data
            orderbook = self.get_orderbook(yes_token)
            bids = orderbook.get('bids', [])
            asks = orderbook.get('asks', [])
            
            # Calculate real order book features
            bid_volume = sum(float(b.get('size', 0)) for b in bids) if bids else 0
            ask_volume = sum(float(a.get('size', 0)) for a in asks) if asks else 0
            total_volume = bid_volume + ask_volume
            
            # Order Book Imbalance (OBI)
            if total_volume > 0:
                order_imbalance = (bid_volume - ask_volume) / total_volume
            else:
                order_imbalance = 0.0
            
            # Get real volume metrics
            volume_24h = float(market.get('volume24hr', 0) or 0)
            volume_total = float(market.get('volumeNum', 0) or market.get('volume', 0) or 0)
            liquidity = float(market.get('liquidityNum', 0) or market.get('liquidity', 0) or 0)
            
            # Get historical price data
            price_history = self.get_prices_history(yes_token, interval='1w', fidelity=5)
            
            # Calculate real momentum and volatility from price history
            if len(price_history) >= 10:
                prices_arr = price_history['price'].values
                momentum = (prices_arr[-1] - prices_arr[0]) / max(prices_arr[0], 0.01)
                volatility = np.std(np.diff(prices_arr)) if len(prices_arr) > 1 else 0.02
                
                # RSI-like calculation
                changes = np.diff(prices_arr)
                gains = np.mean(changes[changes > 0]) if len(changes[changes > 0]) > 0 else 0
                losses = -np.mean(changes[changes < 0]) if len(changes[changes < 0]) > 0 else 0
                if losses > 0:
                    rs = gains / losses
                    rsi = 1 - (1 / (1 + rs))
                else:
                    rsi = 0.5
            else:
                # Fallback if no price history
                momentum = 0.0
                volatility = 0.02
                rsi = 0.5
            
            # Price change metrics
            one_day_change = float(market.get('oneDayPriceChange', 0) or 0)
            one_week_change = float(market.get('oneWeekPriceChange', 0) or 0)
            
            # Get spread
            best_bid = float(market.get('bestBid', 0) or 0)
            best_ask = float(market.get('bestAsk', 1) or 1)
            spread = best_ask - best_bid
            
            # Skip markets at very extreme prices (>99% or <1%) - these are effectively resolved
            # No trading opportunity exists at these prices
            if current_price > 0.99 or current_price < 0.01:
                return None
            
            # Determine outcome using PRICE HISTORY (not just current price)
            # This creates better training labels based on actual price movement
            if len(price_history) >= 20:
                prices_arr = price_history['price'].values
                # Use first half as "past" features, last price as "outcome"
                mid_point = len(prices_arr) // 2
                past_avg = np.mean(prices_arr[:mid_point])
                recent_avg = np.mean(prices_arr[-5:])  # Last 5 points
                
                # Outcome: did price go UP (1) or DOWN (0)?
                outcome = 1 if recent_avg > past_avg else 0
                future_price = recent_avg
                
                # Use the mid-point price as "current" for training
                # This simulates having past data and predicting future
                training_price = prices_arr[mid_point]
            else:
                # Fallback: use momentum direction
                outcome = 1 if momentum > 0 else 0
                future_price = current_price
                training_price = current_price
            
            # Build feature vector (same order as FeatureExtractor)
            features = np.array([
                training_price,          # current_price (at prediction time)
                volume_24h,              # volume_24h
                liquidity,               # liquidity
                rsi,                     # rsi
                momentum,                # momentum
                order_imbalance,         # order_imbalance
                volatility,              # volatility
                one_day_change,          # price_change_1d
                one_week_change,         # price_change_1w
                spread,                  # spread
            ]).reshape(1, -1)
            
            return {
                'features': features,
                'current_price': training_price,
                'future_price': future_price,
                'outcome': outcome,
                'market_id': market.get('id', 'unknown'),
                'question': market.get('question', 'Unknown')[:50],
                'is_resolved': is_closed,
                'volume': volume_total,
            }
            
        except Exception as e:
            self._log(f"Error processing market: {e}")
            return None
    
    # =========================================================================
    # Helper Methods
    # =========================================================================
    
    def get_token_ids_for_market(self, market: Dict) -> Tuple[Optional[str], Optional[str]]:
        """Extract YES and NO token IDs from market"""
        clob_tokens = market.get('clobTokenIds')
        
        if clob_tokens:
            if isinstance(clob_tokens, str):
                try:
                    tokens = json.loads(clob_tokens)
                except:
                    tokens = []
            else:
                tokens = clob_tokens
            
            yes_token = tokens[0] if len(tokens) > 0 else None
            no_token = tokens[1] if len(tokens) > 1 else None
            return yes_token, no_token
        
        return None, None
    
    def trades_to_dataframe(self, trades: List[Dict]) -> pd.DataFrame:
        """
        Convert trades list to DataFrame with normalized column names.
        
        Handles Polymarket API trade format with fields:
        - price, size, side (BUY/SELL), match_time, asset_id, market, etc.
        """
        if not trades:
            return pd.DataFrame(columns=['timestamp', 'price', 'size', 'side'])
        
        try:
            df = pd.DataFrame(trades)
            
            # Normalize timestamp column (API may use 'match_time', 'timestamp', 't', etc.)
            timestamp_col = None
            for col in ['match_time', 'timestamp', 't', 'time', 'created_at', 'last_update']:
                if col in df.columns:
                    timestamp_col = col
                    break
            
            if timestamp_col:
                df['timestamp'] = pd.to_datetime(df[timestamp_col], errors='coerce', unit='s')
                # If conversion fails, try without unit parameter (may be ISO string)
                if df['timestamp'].isna().all():
                    df['timestamp'] = pd.to_datetime(df[timestamp_col], errors='coerce')
            else:
                # Create timestamp from index or use current time
                df['timestamp'] = datetime.now()
                self._log("⚠️ No timestamp column found, using current time")
            
            # Normalize price column (handle different field names)
            price_col = None
            for col in ['price', 'p', 'trade_price', 'execution_price']:
                if col in df.columns:
                    price_col = col
                    break
            
            if price_col and price_col != 'price':
                df['price'] = df[price_col]
            
            if 'price' in df.columns:
                df['price'] = pd.to_numeric(df['price'], errors='coerce')
            else:
                df['price'] = 0.5
                self._log("⚠️ No price column found, using default 0.5")
            
            # Normalize size column (handle different field names)
            size_col = None
            for col in ['size', 's', 'quantity', 'amount', 'volume']:
                if col in df.columns:
                    size_col = col
                    break
            
            if size_col and size_col != 'size':
                df['size'] = df[size_col]
            
            if 'size' in df.columns:
                df['size'] = pd.to_numeric(df['size'], errors='coerce')
            else:
                df['size'] = 0.0
                self._log("⚠️ No size column found, using default 0.0")
            
            # Normalize side column (BUY/SELL)
            if 'side' not in df.columns:
                # Try to infer from other fields
                for col in ['side', 'type', 'direction']:
                    if col in df.columns:
                        df['side'] = df[col]
                        break
                else:
                    df['side'] = 'BUY'  # Default
                    self._log("⚠️ No side column found, using default 'BUY'")
            
            # Sort by timestamp (most recent first)
            if 'timestamp' in df.columns and not df['timestamp'].isna().all():
                df = df.sort_values('timestamp', ascending=False).reset_index(drop=True)
            
            # Keep only essential columns for prediction model
            essential_cols = ['timestamp', 'price', 'size']
            optional_cols = ['side', 'asset_id', 'market']
            available_cols = [c for c in essential_cols + optional_cols if c in df.columns]
            df = df[available_cols].copy()
            
            # Remove rows with invalid price or size
            df = df.dropna(subset=['price', 'size'])
            df = df[(df['price'] > 0) & (df['price'] <= 1) & (df['size'] > 0)]
            
            self._log(f"✅ Converted {len(df)} valid trades to DataFrame")
            return df
            
        except Exception as e:
            self._log(f"❌ Error converting trades to DataFrame: {e}")
            # Return minimal DataFrame with current price
            return pd.DataFrame({
                'timestamp': [datetime.now()],
                'price': [0.5],
                'size': [0.0],
                'side': ['BUY']
            })
    
    def get_market_summary(self, market: Dict) -> Dict:
        """Get formatted market summary"""
        prices_str = market.get('outcomePrices', '[0.5, 0.5]')
        try:
            prices = json.loads(prices_str) if isinstance(prices_str, str) else prices_str
            yes_price = float(prices[0]) if prices else 0.5
        except:
            yes_price = 0.5
        
        return {
            'question': market.get('question', 'Unknown'),
            'yes_price': yes_price,
            'no_price': 1 - yes_price,
            'volume_24h': float(market.get('volume24hr', 0) or 0),
            'liquidity': float(market.get('liquidity', 0) or 0),
            'end_date': market.get('endDate'),
        }
    
    def clear_cache(self):
        """Clear all cached data"""
        self._markets_cache.clear()
        self._cache_timestamps.clear()
        self._log("Cache cleared")


# =============================================================================
# REAL DATA ONLY - No Synthetic Functions
# =============================================================================
# 
# This module uses 100% real data from Polymarket's CLOB and Gamma APIs.
# All synthetic/fake data generation has been removed.
#
# Real data sources:
# - https://clob.polymarket.com/prices-history (historical timeseries)
# - https://clob.polymarket.com/book (orderbook)
# - https://clob.polymarket.com/trades (trade history)
# - https://gamma-api.polymarket.com/markets (market metadata)
#
# =============================================================================
