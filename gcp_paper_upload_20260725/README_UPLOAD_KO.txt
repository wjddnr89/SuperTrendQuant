SuperTrendQuant Google Cloud 가상계좌 업로드 폴더
=================================================

이 폴더 하나만 Google Cloud VM에 올리면 됩니다.

포함된 것
- canonical Nasdaq-100 전략과 가상계좌 설정
- 토스증권 실시간 시세 연동 코드
- 텔레그램 시작, 체결, 일일 요약, 오류 알림
- 기존 검증에 사용한 로컬 시장 데이터 캐시
- Python 의존성 목록
- 매일 데이터 갱신 후 가상계좌를 실행하는 run_paper_daily.sh

실제 주문 안전장치
- execution.broker는 paper입니다.
- 토스증권은 실시간 시세 조회에만 사용합니다.
- 매수와 매도는 로컬 PaperBroker 가상계좌에만 기록됩니다.

이 폴더에 포함하지 않은 비밀정보
- TOSS_CLIENT_ID
- TOSS_CLIENT_SECRET
- TOSS_ACCOUNT_SEQ
- TELEGRAM_TOKEN
- TELEGRAM_CHAT_ID
- EODHD_API_TOKEN
- SEC_USER_AGENT

위 값은 Google Cloud 서버 안에서 별도로 입력해야 합니다.
.env.example은 항목 이름만 보여 주는 빈 양식이며 실제 키는 없습니다.

서버 운용 방식
- e2-micro에서는 프로그램을 1분마다 계속 돌리기보다 하루 한 번 실행하는 방식이 적합합니다.
- 미국 장 마감 후 데이터가 갱신되는 시간에 run_paper_daily.sh가 실행되도록 예약합니다.
- 스크립트는 먼저 새 데이터를 동기화하고, 성공한 경우에만 canonical 가상계좌를 한 번 구동합니다.
- 계좌 상태는 state 폴더에, 실행 기록은 results 폴더에 계속 누적됩니다.

주의
- data/cache 폴더는 삭제하지 마세요. 기존 데이터와 유니버스 검증 근거가 들어 있습니다.
- 서버에 실제 비밀키를 입력한 뒤 이 폴더를 다시 압축하거나 공유하지 마세요.
- 가상계좌 검증이 끝날 때까지 execution.broker를 toss로 바꾸지 마세요.
