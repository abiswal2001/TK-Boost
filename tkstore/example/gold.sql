
SELECT 
    AVG(
        ROUND(ABS(CAST(strftime('%Y', final_game) AS INTEGER) - CAST(strftime('%Y', debut) AS INTEGER)), 2) +
        ROUND(ABS(CAST(strftime('%m', final_game) AS INTEGER) - CAST(strftime('%m', debut) AS INTEGER)) / 12.0, 2) +
        ROUND(ABS(CAST(strftime('%d', final_game) AS INTEGER) - CAST(strftime('%d', debut) AS INTEGER)) / 365.0, 2)
    ) as average_career_span
FROM player 
WHERE debut IS NOT NULL AND final_game IS NOT NULL
