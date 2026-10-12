import calendar
import requests
import pandas as pd


class TradeService:

    def __init__(self):
        self.session = requests.Session()
        self.base_url = 'http://trade-streaming.ext.g20api.com:8767'
    
    def __del__(self):
        self.session.close()

    def _request(self, method, url, **kwargs):
        try:
            response = self.session.request(method, url, **kwargs)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f'Error: {e}')
            return None
        
    def _get_trades(self, **params):
        endpoint = '/trades/'
        response =  self._request('get', self.base_url + endpoint, params=params)
        return response
        
    def get_trades(self, pair, start_dt, end_dt, location_ids=None):
        trades = []
        min_clock_id = None
        while True:
            response = self._get_trades(
                inst = pair,
                min_exch_ts = int(calendar.timegm(start_dt.timetuple()) * 1e6),
                max_exch_ts = int(calendar.timegm(end_dt.timetuple()) * 1e6),
                location_id = location_ids,
                max_clock_id = min_clock_id,
                order_by = 'clock_id'
            )
            if not response:
                break
            trades += response
            min_clock_id = trades[-1]['clock_id']
        trades = pd.DataFrame(trades)
        if trades.empty:
            return trades
        trades['exch_ts'] = pd.to_datetime(trades['exch_ts'], unit="us")
        trades['recv_ts'] = pd.to_datetime(trades['recv_ts'], unit="us")
        trades[['notional','qty','price','fee']] = trades[['notional','qty','price','fee']].astype(float)
        return trades

    def get_all_trades(self, start_dt, end_dt, location_ids=None):
        """Every inst's trades in the window in ONE paginated stream (no inst filter), optionally
        scoped to location_ids. Same shaping as get_trades — a desk-level caller pulls once and
        splits by inst locally, instead of one paginated pull per pair."""
        trades = []
        min_clock_id = None
        while True:
            response = self._get_trades(
                min_exch_ts = int(calendar.timegm(start_dt.timetuple()) * 1e6),
                max_exch_ts = int(calendar.timegm(end_dt.timetuple()) * 1e6),
                location_id = location_ids,
                max_clock_id = min_clock_id,
                order_by = 'clock_id'
            )
            if not response:
                break
            trades += response
            min_clock_id = trades[-1]['clock_id']
        trades = pd.DataFrame(trades)
        if trades.empty:
            return trades
        trades['exch_ts'] = pd.to_datetime(trades['exch_ts'], unit="us")
        trades['recv_ts'] = pd.to_datetime(trades['recv_ts'], unit="us")
        trades[['notional','qty','price','fee']] = trades[['notional','qty','price','fee']].astype(float)
        return trades


class ReferenceService:

    def __init__(self, api_token):
        self.session = requests.Session()
        self.base_url = 'https://refsvc.prod.g20api.com/ref-svc'
        self.session.headers.update({
            'Authorization': f'Bearer {api_token}',
            'Content-Type': 'application/json'
        })
    
    def __del__(self):
        self.session.close()

    def _request(self, method, url, **kwargs):
        try:
            response = self.session.request(method, url, **kwargs)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f'Error: {e}')
            return None
        
    def get_all_locations(self):
        endpoint = '/location'
        response =  self._request('get', self.base_url + endpoint)
        return response
    
    def get_location(self, location_id):
        endpoint = f'/location/uid/{location_id}'
        response =  self._request('get', self.base_url + endpoint)
        return response
