import requests
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TossAuthManager")

class TossAccountManager:
    def __init__(self, client_id: str, client_secret: str):
        self.base_url = "https://openapi.tossinvest.com"
        self.client_id = client_id
        self.client_secret = client_secret
        self.access_token = None
        self.account_seq = None
        
        # 봇 메모리에 유지될 실시간 자산 데이터 세트
        self.krw_cash = 0.0
        self.usd_cash = 0.0
        self.active_positions = {}  # 예: {"005930.KS": {"qty": 10, "avg_price": 72000}}

    def fetch_access_token(self) -> bool:
        """1. OAuth 2.0 Client Credentials 방식으로 토큰을 발급받습니다."""
        url = f"{self.base_url}/oauth2/token"
        headers = {"Content-Type": "application/x-www-form-urlencoded"}
        payload = {
            "grant_type": "client_credentials",
            "client_id": self.client_id,
            "client_secret": self.client_secret
        }
        
        try:
            response = requests.post(url, headers=headers, data=payload)
            if response.status_code == 200:
                data = response.json()
                self.access_token = data.get("access_token")
                logger.info("👉 토스 API 액세스 토큰 발급 성공")
                return True
            else:
                logger.error(f"❌ 토큰 발급 실패: {response.status_code} - {response.text}")
                return False
        except Exception as e:
            logger.error(f"❌ 토큰 요청 중 에러 발생: {e}")
            return False

    def fetch_account_sequence(self) -> bool:
        """2. 종합 매매가 가능한 활성화된(ACTIVE) 계좌의 일련번호를 획득합니다."""
        if not self.access_token:
            logger.error("❌ 액세스 토큰이 없습니다.")
            return False
            
        url = f"{self.base_url}/api/v1/accounts"
        headers = {"Authorization": f"Bearer {self.access_token}"}
        
        try:
            response = requests.get(url, headers=headers)
            if response.status_code == 200:
                accounts = response.json().get("result", [])
                
                # 실전 안전장치: ACTIVE 상태인 첫 번째 종합매매계좌 선택
                for acc in accounts:
                    if acc.get("accountStatus") == "ACTIVE":
                        self.account_seq = acc.get("accountSeq")
                        logger.info(f"👉 사용 가능 실전 계좌 연결 완료 (accountSeq: {self.account_seq})")
                        return True
                
                logger.warning("⚠️ 활성화된(ACTIVE) 종합매매 계좌를 찾을 수 없습니다.")
                return False
            else:
                logger.error(f"❌ 계좌 조회 실패: {response.text}")
                return False
        except Exception as e:
            logger.error(f"❌ 계좌 조회 중 에러 발생: {e}")
            return False

    def sync_buying_power(self) -> bool:
        """3. 원화 및 달러의 실시간 매수 가능 금액(예수금)을 안전하게 동기화합니다."""
        if not self.account_seq:
            return False
            
        url = f"{self.base_url}/api/v1/buying-power"
        headers = {
            "Authorization": f"Bearer {self.access_token}",
            "X-Tossinvest-Account": str(self.account_seq)
        }
        
        try:
            # 원화(KRW) 예수금 조회
            res_krw = requests.get(url, headers=headers, params={"currency": "KRW"})
            if res_krw.status_code == 200:
                # 문자열 숫자가 들어와도 에러 없이 처리하도록 float 캐스팅 안전장치
                raw_cash = res_krw.json().get("result", {}).get("cashBuyingPower", "0")
                self.krw_cash = float(raw_cash)
                
            # 달러(USD) 예수금 조회
            res_usd = requests.get(url, headers=headers, params={"currency": "USD"})
            if res_usd.status_code == 200:
                raw_cash = res_usd.json().get("result", {}).get("cashBuyingPower", "0")
                self.usd_cash = float(raw_cash)
                
            logger.info(f"👉 예수금 동기화 완료 | KRW: {self.krw_cash:,.0f}원 | USD: ${self.usd_cash:,.2f}")
            return True
        except Exception as e:
            logger.error(f"❌ 예수금 동기화 중 에러 발생: {e}")
            return False

    def sync_holdings(self) -> bool:
        """4. 현재 보유 종목을 긁어와 universe.json 규격(.KS/.KQ)과 호환되도록 매핑합니다."""
        if not self.account_seq:
            return False
            
        url = f"{self.base_url}/api/v1/holdings"
        headers = {
            "Authorization": f"Bearer {self.access_token}",
            "X-Tossinvest-Account": str(self.account_seq)
        }
        
        try:
            response = requests.get(url, headers=headers)
            if response.status_code == 200:
                holdings_data = response.json().get("result", {}).get("items", [])
                
                updated_positions = {}
                for item in holdings_data:
                    symbol = item.get("symbol", "")
                    qty = int(float(item.get("quantity", "0")))  # 안정적인 정수 변환
                    avg_price = float(item.get("averagePurchasePrice", "0.0"))
                    
                    if qty > 0:
                        # 💡 universe.json의 포맷과 맞추기 위한 주식 시장 접미사 매핑
                        if symbol.isdigit():  # 한국 주식 (숫자로만 구성됨)
                            # 우선 코스피(.KS)를 기본값으로 두고, 필요시 전략단에서 유연하게 체크하도록 처리
                            # (보수적인 매칭을 원하시면 봇 메인의 universe_map 키값과 직접 연동도 가능합니다)
                            formatted_symbol = f"{symbol}.KS" 
                        else:  # 미국 주식 티커 (AAPL, QQQ 등)
                            formatted_symbol = symbol
                            
                        updated_positions[formatted_symbol] = {"qty": qty, "avg_price": avg_price}
                        
                self.active_positions = updated_positions
                logger.info(f"👉 보유 종목 동기화 완료 (총 {len(self.active_positions)}개 종목 보유 중)")
                return True
            else:
                logger.error(f"❌ 보유 주식 조회 실패: {response.text}")
                return False
        except Exception as e:
            logger.error(f"❌ 보유 주식 동기화 중 에러 발생: {e}")
            return False

    def initialize_pipeline(self) -> bool:
        """모든 데이터 동기화 파이프라인을 순차적으로 가동합니다."""
        if self.fetch_access_token():
            if self.fetch_account_sequence():
                if self.sync_buying_power() and self.sync_holdings():
                    logger.info("🔥 토스증권 자산 엔진 초기화 최종 완료!")
                    return True
        return False