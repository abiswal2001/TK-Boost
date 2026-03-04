
WITH track_revenue AS (
    SELECT 
        t.TrackId,
        t.Milliseconds,
        t.Milliseconds / 60000.0 AS duration_minutes,
        CASE 
            WHEN t.Milliseconds <= 197335 THEN 'Short'
            WHEN t.Milliseconds > 197335 AND t.Milliseconds <= 2840276 THEN 'Medium'
            ELSE 'Long'
        END AS length_category,
        COALESCE(SUM(il.UnitPrice * il.Quantity), 0) AS track_revenue
    FROM Track t
    LEFT JOIN InvoiceLine il ON t.TrackId = il.TrackId
    GROUP BY t.TrackId, t.Milliseconds
),
category_stats AS (
    SELECT 
        length_category,
        MIN(duration_minutes) AS min_minutes,
        MAX(duration_minutes) AS max_minutes,
        SUM(track_revenue) AS total_revenue
    FROM track_revenue
    GROUP BY length_category
)
SELECT 
    ROUND(min_minutes, 4) AS From_Minutes,
    ROUND(max_minutes, 4) AS To_Minutes,
    length_category AS LengthCateg,
    ROUND(total_revenue, 4) AS TotalPrice
FROM category_stats
ORDER BY 
    CASE length_category 
        WHEN 'Short' THEN 1 
        WHEN 'Medium' THEN 2 
        WHEN 'Long' THEN 3 
    END,
    length_category
