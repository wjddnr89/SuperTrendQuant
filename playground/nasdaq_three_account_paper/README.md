# Nasdaq-100 A~G 가상계좌

`unified_quant`를 수정하지 않고 나스닥 전략을 매일 가상 체결하고 일일·주간 성과를 기록하는 playground 실험입니다.

## 계좌 순서와 규칙

| 계좌 | 표시 이름 | 시장필터 | 추가 규칙 |
|---|---|---|---|
| A | MF ON - Base | QQQ 일봉 | 기본형 |
| B | MF ON - Stop12 GateOff | QQQ 일봉 | 고정손절 12%, 회전수익 gate 해제 |
| C | MF ON - ATR2.5 | QQQ 일봉 | ATR 위험예산 2.5% |
| D | MF OFF - Base | 없음 | 기본형 |
| E | MF OFF - ATR2.5 | 없음 | ATR 위험예산 2.5% |
| F | MF OFF - ATR2.0 | 없음 | ATR 위험예산 2.0% |
| G | MF OFF - 2h Exit | 없음 | 2시간봉 장중 청산, 1분봉 10분 확인 |

공통 규칙은 `dual_momentum/150`, `Ichimoku + EMA trend`, sell confirm `1`, rotation hurdle `2.0`, SuperTrend `10/3.0`, 최대 보유 `1`종목입니다. 수수료는 `0.001`, 슬리피지는 `0.0005`입니다.

기본형은 회전 시 기존 종목 손익이 0% 이상이어야 하며 고정손절과 late-chase 이격 제한이 없습니다. SuperTrend 하락 청산은 회전수익 gate와 무관합니다.

ATR 계좌의 신규·교체 매수비중은 다음과 같습니다.

```text
목표 비중 = min(100%, entry_atr_risk_pct / ATR_pct)
ATR_pct = Wilder ATR(10) / 종가
```

ATR은 매수수량만 조절하며 손절선이 아닙니다. 보유 중 ATR 변화에 따른 일일 리밸런싱은 하지 않습니다.

G는 완성된 2시간봉 SuperTrend 울타리와 1분봉을 재생합니다. 울타리를 10분 연속 이탈하면 다음 1분봉 시가로 가상 매도하고, 완성된 2시간봉 상승 회복 전까지 같은 종목의 재진입을 차단합니다.

## 실행

저장소 루트에서:

```powershell
playground\nasdaq_three_account_paper\run_daily.ps1
playground\nasdaq_three_account_paper\run_weekly.ps1
```

Windows 예약 작업:

- 일일: 매일 06:30 KST, `run_daily.ps1`
- 주간: 일요일 07:00 KST, `run_weekly.ps1`
- 예약 시각에 PC가 꺼져 있으면 로그인 후 가능한 시점에 실행

등록:

```powershell
powershell -ExecutionPolicy Bypass -File playground\nasdaq_three_account_paper\install_windows_tasks.ps1
```

## 결과 위치

```text
state/accounts/A.json ... G.json   계좌별 현금·보유상태
results/daily_history.csv          전체 일일 성과
results/daily/YYYY-MM-DD.csv       날짜별 A~G 비교
results/weekly_history.csv         전체 주간 성과
results/weekly/YYYY-Www.csv        주간 A~G 비교
results/events/                    주문·체결 이벤트
results/dashboard.html             최신 대시보드
logs/                              예약 작업 실행 로그
```

2026-08-04 계좌 재정렬 전 상태와 보고서는 `archive/account_order_before_v2_20260804_234559`에 보존되어 있습니다.
