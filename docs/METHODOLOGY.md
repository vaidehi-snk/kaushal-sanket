# Demand index and forecasting method (run run-20261012-s11, SYNTHETIC data)
1. **Sources**: job_portal, industry_hiring, ncs_vacancies, eshram_registrations. Each is normalised per trade and district with median and MAD (robust to outliers).
2. **Learned weights**: non-negative ridge regression predicts next month's placements from this month's four normalised signals, pooled over 42 complete series (R2 0.43).
   Weights: job_portal 23% (+/-4), industry_hiring 44% (+/-5), ncs_vacancies 17% (+/-5), eshram_registrations 15% (+/-3). Leaving a source out costs R2: job_portal 0.015, industry_hiring 0.041, ncs_vacancies 0.007, eshram_registrations 0.010.
3. **Missing source**: weights of the remaining sources are renormalised for that series.
4. **Scale**: the index is expressed in hires per month using the series' own placements median and MAD.
5. **Forecast**: seasonal-naive vs ETS, chosen per series by rolling-origin backtest (MASE). 6-month horizon, 80% range from backtest RMSE (independence assumed, an approximation).
6. **Call rule**: shortage or oversupply only if the whole 80% range clears +/-0.15. History under 33 months or MASE above 1.5: no call.
7. **Early warning**: central forecast crosses +/-0.15 within 6 months while today's call is balanced or uncertain.
8. **Seat target**: 12 x forecast monthly demand / completion rate (0.8). Change = target - seats now, only when a call is made.
9. **Reconciliation**: every level (national, state, district, sector, trade and their combinations) is forecast on its own, then reconciled (structural weighted least squares; MinT with shrinkage and OLS are compared on a held-out test) so all levels add up exactly. Intervals use the reconciled error covariance with shrunk residual correlations. Series that abstained are left out of roll-ups, never imputed.
10. **Validation**: lead-time test, 6 months ahead, recall 0.93, false-alarm rate 0.04 on 58 series (synthetic, so a plumbing check, not an accuracy claim).
11. **Likelihood words**: P(gap beyond +/-0.15) from the forecast error, reported with IPCC AR5 wording (likely >= 66%, very likely >= 90%, virtually certain >= 99%). The call rule (whole 80% range beyond the threshold) equals "at least 90% likely", so words and calls agree.
12. **Levers (what to do, not only where)**: from the supply funnel. Placement of certified trainees under 50% -> review the course with the SSC before adding seats. Shortage with seats under 70% filled -> fill empty seats first (no seat increase). Shortage otherwise -> add capacity. Oversupply with seats under 70% filled -> shrink or merge; otherwise -> redirect seats to a trade in shortage.
13. **QP register**: every trade maps to an NSQF Qualification Pack read from the NSDC QP file (data/qp_register.csv: code, NCO-2015, level, status). A deactivated QP is flagged in every recommendation: no new batches on it, move seats to an active QP.
