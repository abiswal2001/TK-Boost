WITH bowler_runs AS (
    -- Runs scored off the bat (attributed to bowler)
    SELECT 
        bb.bowler,
        SUM(bs.runs_scored) as runs_from_bat
    FROM ball_by_ball bb
    JOIN batsman_scored bs ON bb.match_id = bs.match_id 
                           AND bb.over_id = bs.over_id 
                           AND bb.ball_id = bs.ball_id 
                           AND bb.innings_no = bs.innings_no
    GROUP BY bb.bowler
    
    UNION ALL
    
    -- Extra runs that are attributed to bowler (wides and noballs)
    SELECT 
        bb.bowler,
        SUM(er.extra_runs) as runs_from_extras
    FROM ball_by_ball bb
    JOIN extra_runs er ON bb.match_id = er.match_id 
                       AND bb.over_id = er.over_id 
                       AND bb.ball_id = er.ball_id 
                       AND bb.innings_no = er.innings_no
    WHERE er.extra_type IN ('wides', 'noballs')
    GROUP BY bb.bowler
),
total_runs_conceded AS (
    SELECT 
        bowler,
        SUM(runs_from_bat) as total_runs_conceded
    FROM bowler_runs
    GROUP BY bowler
),
total_wickets AS (
    SELECT 
        bb.bowler,
        COUNT(*) as total_wickets_taken
    FROM ball_by_ball bb
    JOIN wicket_taken wt ON bb.match_id = wt.match_id 
                         AND bb.over_id = wt.over_id 
                         AND bb.ball_id = wt.ball_id 
                         AND bb.innings_no = wt.innings_no
    GROUP BY bb.bowler
),
bowling_averages AS (
    SELECT 
        p.player_name,
        tr.bowler,
        tr.total_runs_conceded,
        tw.total_wickets_taken,
        CAST(tr.total_runs_conceded AS FLOAT) / tw.total_wickets_taken as bowling_average
    FROM total_runs_conceded tr
    JOIN total_wickets tw ON tr.bowler = tw.bowler
    JOIN player p ON tr.bowler = p.player_id
    WHERE tw.total_wickets_taken > 0  -- Only include bowlers who have taken wickets
)
SELECT 
    player_name,
    bowler,
    total_runs_conceded,
    total_wickets_taken,
    ROUND(bowling_average, 4) as bowling_average
FROM bowling_averages
ORDER BY bowling_average ASC, player_name ASC
LIMIT 1