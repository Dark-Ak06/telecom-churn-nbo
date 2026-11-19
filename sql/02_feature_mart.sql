WITH monthly AS (
    SELECT * FROM read_parquet($raw_path)
), windows AS (
    SELECT
        *,
        LAG(month) OVER subscriber AS previous_observation_month,
        LAG(data_gb) OVER subscriber AS previous_data_gb,
        LAG(app_sessions) OVER subscriber AS previous_app_sessions,
        AVG(data_gb) OVER trailing_three AS data_gb_3m_avg,
        AVG(arpu_kzt) OVER trailing_three AS arpu_3m_avg,
        SUM(support_calls) OVER (
            PARTITION BY user_id ORDER BY month ROWS UNBOUNDED PRECEDING
        ) AS support_calls_cum
    FROM monthly
    WINDOW
        subscriber AS (PARTITION BY user_id ORDER BY month),
        trailing_three AS (
            PARTITION BY user_id ORDER BY month
            RANGE BETWEEN INTERVAL 2 MONTHS PRECEDING AND CURRENT ROW
        )
), lagged AS (
    SELECT *,
        COALESCE(previous_observation_month = month - INTERVAL 1 MONTH, FALSE) AS has_previous_month
    FROM windows
)
SELECT
    * EXCLUDE (previous_observation_month, previous_data_gb, previous_app_sessions),
    CASE WHEN has_previous_month AND previous_data_gb <> 0
        THEN 100.0 * (data_gb - previous_data_gb) / previous_data_gb ELSE 0 END
        AS data_gb_mom_change,
    CASE WHEN has_previous_month
        THEN 100.0 * (app_sessions - previous_app_sessions) / GREATEST(previous_app_sessions, 1)
        ELSE 0 END AS app_sessions_mom_change,
    CASE
        WHEN days_since_last_topup >= 32 AND app_sessions <= 5 THEN 'At_Risk'
        WHEN days_since_last_topup <= 14 AND app_sessions >= 15 AND arpu_kzt >= 6500 THEN 'Champions'
        WHEN days_since_last_topup <= 23 AND app_sessions >= 8 THEN 'Engaged'
        ELSE 'Occasional'
    END AS rfm_segment
FROM lagged;
