
WITH motorcycle_helmet_status AS (
    -- Get all motorcycle collisions and their helmet usage status
    SELECT DISTINCT
        c.case_id,
        c.motorcyclist_killed_count,
        CASE 
            WHEN (p.party_safety_equipment_1 LIKE '%motorcycle helmet used%' 
                  OR p.party_safety_equipment_2 LIKE '%motorcycle helmet used%') THEN 'helmet_used'
            WHEN (p.party_safety_equipment_1 LIKE '%motorcycle helmet not used%' 
                  OR p.party_safety_equipment_2 LIKE '%motorcycle helmet not used%') THEN 'helmet_not_used'
            ELSE 'unknown'
        END as helmet_status
    FROM collisions c
    JOIN parties p ON c.case_id = p.case_id
    WHERE c.motorcycle_collision = 1
    AND (p.party_safety_equipment_1 LIKE '%motorcycle%' OR p.party_safety_equipment_2 LIKE '%motorcycle%')
),

collision_helmet_summary AS (
    -- Summarize helmet usage per collision (handle multiple parties per collision)
    SELECT 
        case_id,
        motorcyclist_killed_count,
        CASE 
            WHEN COUNT(CASE WHEN helmet_status = 'helmet_used' THEN 1 END) > 0 THEN 'helmet_used'
            WHEN COUNT(CASE WHEN helmet_status = 'helmet_not_used' THEN 1 END) > 0 THEN 'helmet_not_used'
            ELSE 'unknown'
        END as final_helmet_status
    FROM motorcycle_helmet_status
    WHERE helmet_status IN ('helmet_used', 'helmet_not_used')
    GROUP BY case_id, motorcyclist_killed_count
),

helmet_stats AS (
    -- Calculate totals for each helmet usage group
    SELECT 
        final_helmet_status,
        COUNT(case_id) as total_collisions,
        SUM(motorcyclist_killed_count) as total_fatalities
    FROM collision_helmet_summary
    GROUP BY final_helmet_status
)

-- Calculate final percentages
SELECT 
    ROUND(
        (SELECT total_fatalities * 100.0 / total_collisions 
         FROM helmet_stats 
         WHERE final_helmet_status = 'helmet_used'), 4
    ) as percent_killed_helmet_used,
    ROUND(
        (SELECT total_fatalities * 100.0 / total_collisions 
         FROM helmet_stats 
         WHERE final_helmet_status = 'helmet_not_used'), 4
    ) as percent_killed_helmet_not_used
