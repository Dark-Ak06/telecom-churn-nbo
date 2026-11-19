WITH grouped AS (
    SELECT
        CASE
            WHEN GROUPING(tariff_plan) = 0 THEN 'tariff_plan'
            WHEN GROUPING(region) = 0 THEN 'region'
            WHEN GROUPING(rfm_segment) = 0 THEN 'rfm_segment'
            ELSE 'month'
        END AS dimension,
        COALESCE(tariff_plan, region, rfm_segment, CAST(month AS VARCHAR)) AS segment,
        COUNT(*) AS observations,
        AVG(churn) AS churn_rate,
        AVG(arpu_kzt) AS avg_arpu_kzt
    FROM feature_mart
    GROUP BY GROUPING SETS ((tariff_plan), (region), (rfm_segment), (month))
)
SELECT * FROM grouped ORDER BY dimension, segment;
