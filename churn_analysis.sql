-- Churn rate by contract type
SELECT Contract,
       AVG(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) AS churn_rate
FROM customers
GROUP BY Contract
ORDER BY churn_rate DESC;
