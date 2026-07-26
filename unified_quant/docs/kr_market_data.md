# 한국시장 데이터 운영

## 범위와 불변조건

- 대상은 `KOSPI200 + KOSDAQ150`, 기본 시작일은 `2015-01-01`이다.
- 로그인한 KRX Data Marketplace의 날짜별 과거 구성종목 스냅샷을 모두
  체크포인트로 남겨 당시 유니버스를 재생한다. KOSDAQ150의 실제 지수 출범
  전 구간은 명시적으로 비어 있다. 라이선스된 완전 스냅샷 파일도 선택적
  대체 입력으로 지원한다.
- 종목 ID는 6자리 단축코드가 아니라 KRX ISIN(`KR:<ISIN>`)이다. 단축코드는
  `005930` 같은 숫자형뿐 아니라 `0126Z0` 같은 영숫자형도 허용한다. 현재
  종목과 상장폐지 종목을 함께 수집하고, OpenDART corp code는 보조 식별자로
  붙인다.
- 인증된 KRX Open API 또는 로그인한 Data Marketplace의 동일한 비수정
  OHLCV와 일별 공식 기준가격이 정본이다. 정식 release는 2015년 시작일부터
  종료일까지 각 KRX 거래행의 OHLCV가 KIS·EODHD 중 하나 이상의 독립
  원주가와 일치하는 다중 공급자 hard gate를 통과해야 한다. Naver는 선택적
  fallback으로 추가할 수 있다. 한 공급자의 시장이전 과거 구간 누락은 다른
  공급자가 정확히 확인한 행으로만 보완할 수 있다. 252세션 또는 일부 종목
  비교는 smoke일 뿐 정식 검증으로 승격되지 않는다.
- 동일한 KRX 원문을 다른 이름으로 저장한 데이터와의 값 일치는 독립 검증이
  아니다. 가격은 KIS, 현금배당은 OpenDART 원문, 가격 재산정은 KRX 기준가격,
  지수 구성은 KRX 날짜별 완전 스냅샷으로 서로 다른 증거 계통을 사용한다.
- 로컬 루트는 `data/cache/markets/KR`, R2 prefix는
  `supertrend-quant/markets/KR`이며 US release와 포인터를 공유하지 않는다.

## 1. 공급자 품질 비교

먼저 `.env`에 다음 입력을 준비한다.

```dotenv
KRX_OPENAPI_AUTH_KEY=...
KRX_DAILY_PRICE_TRANSPORT=auto
KRX_ID=...
KRX_PW="..."
KRX_SESSION_CACHE_PATH=data/cache/private/krx_web_session.json
# 선택적 파일 대체 입력
KRX_PIT_CONSTITUENTS_PATH=data/imports/krx/index_constituents.parquet
```

`KRX_OPENAPI_AUTH_KEY`는 [KRX Open API](https://openapi.krx.co.kr/)에서
신청한다. 공개 Open API의 주식 일별매매정보는 2010년 이후를 제공하지만
`AUTH_KEY`가 필요하다. 과거 지수 구성종목은 `KRX_ID`/`KRX_PW`로 로그인한
Data Marketplace의 날짜별 지수구성종목 화면에서 받는다. 공개 Open API
가격 키와 웹 로그인은 서로 다른 인증이다. 공식 반영주식수·편입비중까지
필요하면 별도 [KRX 지수정보상품](https://openapi.krx.co.kr/contents/OPP/DATA/OPPDATA005.jsp)을
사용하고, 그 완전 스냅샷 파일을 선택적 대체 입력으로 로컬에 둘 수 있다.
키 발급과 별도로 `유가증권 일별매매정보`, `코스닥 일별매매정보`,
`ETF 일별매매정보`가 승인되어야 한다. `auto`는 Open API를 먼저 쓰고 지연·권한
오류 시 로그인한 공식 웹 원문으로 전환한다. 장기 전시장 일괄 수집은 실측상
`web`이 더 빠르므로 `KRX_DAILY_PRICE_TRANSPORT=web`으로 고정할 수 있다.
웹 로그인 쿠키는 자격증명 해시와 묶어 권한 `0600`인 로컬 캐시에만 저장하고,
장기 수집 재시작 시 재사용한다. 실제 `LOGOUT`이나 인증 HTTP 오류가 확인된
경우에만 다시 로그인하며, 로그인 실패 직후 동시 작업의 연쇄 재시도도 막는다.
어느 경우에도 비공식 보조 공급자로 정본을 대체하지 않는다.

대체 스냅샷 파일은 CSV/JSON/JSONL/Parquet 중 하나이며 다음 네 열이 필수다.

| 열 | 값 |
| --- | --- |
| `profile` | `kospi200` 또는 `kosdaq150` |
| `session` | 스냅샷 효력일 `YYYY-MM-DD` |
| `symbol` | 6자리 숫자/영숫자 단축코드 |
| `isin` | 12자리 KRX 표준코드 |

각 `profile + session`은 일부 변경분이 아니라 완전 구성이어야 한다. 명목상
200개/150개지만 분할 신설 종목의 일시 동시 편입처럼 실제 KRX 원문이 ±1개
달라지는 기간이 있어 검증 범위는 각각 195~205개/145~155개다. 요청일에는 그
날짜 이하의 최신 스냅샷을 재생한다. 키나 파일이 없거나 이 범위를 벗어나거나
ISIN이 틀리면 benchmark, bootstrap, sync는 보조 공급자로 우회하지 않고
즉시 중단한다.

```bash
uv run quant-data benchmark-kr \
  --start 2015-01-01 \
  --providers krx,kis,eodhd \
  --krx-workers 4
```

중단해도 `benchmarks/<기간>/`의 구성종목·가격·원본 증거 체크포인트에서
재개된다. 공급자 결과도 원본 해시가 모두 맞으면 재사용하며, 실제 재수집과
수정 안정성 재측정이 필요할 때만 `--refresh-providers`를 사용한다. 빠른 연결
확인은 `--symbols-limit 5 --sessions 20`을 사용한다.
KIS 키가 없으면 실패로 위장하지 않고 `skipped_missing_credentials`로 기록된다.
가격 비교에는 기본적으로 실전투자(`prod`) App Key/Secret을 사용한다.
KIS 일봉 API의 조회기간 제한 때문에 종목별 100일 이하 조각으로 수집하며,
각 조각의 요청·응답 해시와 Parquet를 저장하므로 장시간 실행도 성공한
조각부터 재개한다.
KIS 접근 토큰은 24시간 유효하고 재발급 제한이 있으므로
`data/cache/private/kis_prod_access_token.json`에 권한 `0600`으로 캐시한다.
모의·실전 계정은 서로 다른 다음 환경 변수로 분리한다.

```dotenv
KIS_MARKET_DATA_MODE=prod
KIS_PROD_APP_KEY=...
KIS_PROD_APP_SECRET=...
KIS_PROD_ACCOUNT_NO=계좌번호앞8자리
KIS_PROD_ACCOUNT_PRODUCT_CODE=01
KIS_PAPER_APP_KEY=...
KIS_PAPER_APP_SECRET=...
KIS_PAPER_ACCOUNT_NO=모의계좌번호앞8자리
KIS_PAPER_ACCOUNT_PRODUCT_CODE=01
```

현재 KIS 경로는 원주가 비교까지만 담당하며 주문은 아직 Toss 경로를 사용한다.
계좌 변수는 후속 KIS 주문 어댑터가 두 환경을 혼동하지 않도록 미리 분리한
계약이다.
스모크 결과는 `benchmarks/current-smoke.json`에 따로 기록되며 상태도
`smoke_ready`다. 전체 PIT 합집합을 검사한 `benchmarks/current.json`만 정식
백필을 승인하므로, 스모크 실행이 기존 정식 승인을 덮어쓰지 않는다.

정식 membership gate는 KOSPI200 2015-01-02 이후와 KOSDAQ150
2015-07-13 출범 이후의 모든 XKRX 세션을 검사한다. 2026-07-22 경계에서는
각각 2,835개와 2,705개, 합계 5,540개 완전 스냅샷을 anchor와 ADD/REMOVE
이벤트로 다시 재생해 매일의 집합이 원문과 정확히 같은지 확인한다. 일부
대표일이나 현재 구성종목만 맞는 것으로는 이 gate를 통과할 수 없다.

개별 공급자 하드 게이트는 ISIN 매핑 100%, 중복/비정상 OHLC 0건, KRX 기대
행 커버리지 100%, KRX 한 틱 이내 OHLC 100%, 거래량 일치 100%
이상, 미분류 누락 0건, 공급자에만 나타나는 35% 이상 시계열 단절 0건, 모든
정규화 행의 요청 원본 SHA-256 연결 100%다. 다중 공급자 게이트는 더 엄격하게
모든 KRX 거래행에 대해 적어도 한 공급자의 OHLC가 한 틱 이내이고 거래량이
정확히 같아야 하며, 어느 공급자도 확인하지 못한 행은 1건만 있어도 차단한다.
KRX에도 같은 방향·크기로 나타나는 분할이나 실제 급등락은 설명된 변동으로
분류한다. 통과 공급자는 정확도 35%, 커버리지 20%, 기업행동 15%,
PIT 생존편향 15%, 반복 실행 간 수정 안정성 10%, 상대 운영비용 5%로 순위를
매긴다. 최초 실행의 수정 안정성은 중립값이고, 같은 기간을 다시 실행하면
기존 체크포인트와 겹치는 raw OHLCV의 수정률로 교체된다. KRX 정본과 이
게이트를 통과한 다중 공급자 합성만 전체 이력의 secondary가 될 수 있으며,
단일 공급자만 통과한 결과는 정식 `ready`가 되지 않는다.
KRX 응답의 거래량 0인 행은 OHLCV bar를 만들지 않고
`suspended_or_no_trade` 관측으로 체크포인트에 남긴다. 반면 활성 종목이 공식
응답에서 사라지거나 비정상 OHLC를 반환하면 `unclassified`로 남아 게이트를
차단한다. 단, KRX 상장폐지 목록의 효력일에 종목이 가격 응답에서 사라진
경우는 `delisting_effective_date_no_trade`로 명시 분류한다.
보조 공급자의 거래량 0 bar도 같은 `suspended_or_no_trade`로 분류한다.
공식 KRX 관측에 없는 날짜를 bar로 만들거나, KRX가 매매정지로 분류한 날짜를
실제 체결 bar로 반환하면 각각 `unexpected_provider_observations`,
`misclassified_official_no_trade_observations`로 집계한다. 해당 단일 공급자의
hard gate는 차단한다. 다중 공급자 합성에서는 이런 공급자 전용 행을 canonical
표에 넣지 않고 격리하며, 반대로 모든 공식 KRX 거래행을 적어도 한 공급자가
정확히 확인했는지를 기준으로 release를 차단한다.

설치된 `exchange_calendars`보다 늦게 확정된 2026-06-03 지방선거일과
2026-07-17 제헌절 공휴일은 출처 URL과 함께 release metadata의
`calendar_closure_overrides`에 기록한다. 장기 캘린더 감사에서는 거래량 0인
합성 bar를 거래일 근거로 쓰지 않는다.

## 2. 2015년 전체 구축과 검증

```bash
uv run quant-data bootstrap-kr \
  --start 2015-01-01 \
  --krx-workers 4

uv run quant-data validate --market KR
uv run quant-data status --market KR
```

구축 결과는 신원, 심볼 이력, KRX 공식 raw 가격, 기업행동, 조정계수, PIT anchor와
ADD/REMOVE 이벤트, lifecycle resolution, 교차검증 보고서, 허용된 원본 archive를
하나의 immutable release로 묶는다. 검증은 각 Parquet를 한 번만 읽어 스키마,
XKRX 세션, factor coverage, PIT 가격/신원 연결, archive license를 함께 확인한다.
어느 하드 게이트라도 실패하면 release pointer는 전진하지 않는다.

### 기업행동과 lifecycle 증거

EODHD의 배당·분할만으로 합병·주식교환·상장폐지 경제조건을 추정하지 않는다.
모든 KRX 거래행의 전일 종가와 공식 기준가격을 비교해 가격 재산정일마다
`reference_price_adjustment`를 만든다. 공급자의 분할·주식배당 비율은 수량
증거로 남기되, 같은 날 가격계수에는 KRX 공식 기준가격을 한 번만 적용한다.
반대로 KRX 과거 가격열이 액면분할 전 구간까지 이미 소급 재작성돼 전일 종가와
기준가격이 연속이면 비율 1.0의 공식 no-op을 만든다. 따라서 공급자 분할을 다시
적용해 과거 가격을 이중 조정하지 않는다. 모든 공급자 비율 이벤트는 공식
기준가격 조정, 재작성 no-op, 또는 가격계수 적용 구간 밖이라는 세 경우 중
하나로 감사되어야 하며, 처음 발견된 과거 이벤트가 증분 구간 밖에 있으면
동기화를 중단하고 전기간 bootstrap을 요구한다.

현금배당은 scoped 보통주 발행사의 OpenDART 최종 정정공시를 전부 paging하고
1주당 금액·기준일·지급일·결의일과 원문 SHA-256을 보관한다. 배당락일은 KRX의
T+2 규칙과 XKRX 휴장일을 사용해 기준일 이하 두 번째 마지막 거래일로
결정한다. EODHD 배당 원본은 독립 비교 증거로 감사보고서에 남기되 canonical
action에서는 제거하고, OpenDART 공식 금액만 수익률 계수에 적용한다.
공급자에만 있고 완전한 OpenDART 공시 목록에 없는 배당도 거부된 공급자 행으로
감사보고서에 남긴다. 반대로 OpenDART 결정에 대응하는 공식 action 누락,
해석하지 못한 정정공시, canonical 표에 남은 미확인 공급자 배당은 release를
차단한다.

가격이 영구 중단된 과거 구성종목이 있으면 다음 입력 중 하나가 있어야 한다.

```dotenv
KR_OFFICIAL_ACTIONS_PATH=data/imports/krx/official_actions.parquet
KR_LIFECYCLE_RESOLUTIONS_PATH=data/imports/krx/lifecycle_resolutions.parquet
```

`KR_OFFICIAL_ACTIONS_PATH`는 CSV/JSON/JSONL/Parquet이며 최소한
`security_id` 또는 `isin`, `action_type`, `effective_date`, `source_url`을 가진다.
현금합병은 `cash_amount`, 주식합병은 `ratio + new_isin/new_symbol`, 상장폐지는
검증된 `cash_amount`(무상소각이면 `0`)가 필수다. `source_url`은 실제
KRX/KIND/OpenDART HTTP(S) 근거여야 한다. 후속 종목도 security master에
포함되도록 `new_symbol`을 기록한다.

공식 경제조건으로 표현할 수 없는 건만 두 번째 파일에 승인 예외로 기록한다.
필수 열은 `security_id`, `last_price_date`, `exception_code`,
`exception_reason`, `reviewed_by`, `reviewed_at`, `source_url`이다. 임시 예외는
미래 `recheck_after`가 없으면 실패한다. 코드는 단순 상장폐지 사실만 보고
회수금액 `0`을 만들어내지 않으며, 모든 candidate가 적용 action 또는 승인
예외로 닫히기 전에는 release를 만들지 않는다.

## 3. 일일 동기화와 충돌

```bash
uv run quant-data sync --market KR
```

최근 7일을 겹쳐 다시 받고 새 세션만 append한다. 이미 확정된 같은 키의
OHLCV가 달라지면 새 값으로 덮지 않고 `conflicts/`에 격리하며 release를
전진시키지 않는다. 모든 KRX 호출은 세션별 체크포인트라 재시작할 수 있다.
구성 변경이 없는 날도 두 공식 스냅샷을 확인했다는 coverage 종료일을 새
manifest에 기록한다. 새로 검증한 과거 lifecycle 근거는 전체 가격을 다시
받지 않고 action/factor와 downstream 검증만 갱신한다.

검증은 빠른 순서로 실행한다. 먼저 키·스키마·세션·라이선스와 lifecycle
blocker를 로컬에서 확인하고, 각 Parquet를 한 번 읽어 cross-dataset 검사를
재사용한다. 그 다음에만 설정된 R2 비공개 증거 확인과 조건부 업로드를 수행하며,
마지막으로 빈 `/tmp` 캐시에서 원격 bytes/hash/논리 행을 다시 검사한다.
따라서 느린 원격 cold-cache 검증은 값싼 로컬 게이트가 모두 통과한 뒤에만
실행된다.

## 4. 비공개 R2 게시

```bash
PYTHONPATH=unified_quant/src .venv/bin/python \
  unified_quant/scripts/publish_and_verify_r2.py \
  --market KR --preflight-only

PYTHONPATH=unified_quant/src .venv/bin/python \
  unified_quant/scripts/publish_and_verify_r2.py --market KR
```

KRX/Naver/Yahoo/KIS/OpenDART 원본은 `local_only`라 R2에 올리지 않는다.
정규화 Parquet, EODHD private-use 원본, 품질 보고서만 코드에 고정된
`allowed_private` 정책으로 게시된다. 게시 전 Cloudflare 비공개 상태를 확인하고,
조건부 쓰기 후 `/tmp`의 빈 캐시로 다시 내려받아 release·manifest·Parquet·원본
해시를 재검증한다.
온라인 비공개 상태 확인에는 `.env`의 `CLOUDFLARE_API_TOKEN`이 필요하며,
토큰에는 대상 계정의 `Workers R2 Storage Read` 권한만 부여한다.

## 5. 연구에서 실매매로 승격

연구/백테스트는 `research_kr.yaml`, 실매매는 `live_toss_kr.yaml`을 사용한다.
두 런타임 모두 같은 `index_events`와 같은 strategy YAML을 사용한다. 실매매는
D일 확정 release만 허용하고 D+1 XKRX 개장 후 15분 동안만 주문한다. 주문 전
signal session, data version, strategy hash, order intents를 immutable plan으로
저장하며, append-only 주문 원장·브로커 미체결 주문·계좌 수량을 매번 대조한다.
`state/live/kr/KILL_SWITCH` 파일이 있거나 일손실 한도를 넘거나 데이터가
degraded/blocked이면 신규 주문을 차단한다.
