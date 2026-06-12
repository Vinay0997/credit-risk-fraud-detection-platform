-- Example Snowflake feature views for production alignment.
-- Replace database, schema, and table names with approved enterprise objects.

create or replace view RISK_ANALYTICS.FEATURES.CREDIT_RISK_FEATURES as
select
    c.customer_id,
    t.transaction_id,
    t.event_timestamp,
    c.age,
    c.annual_income,
    c.employment_years,
    b.bureau_score,
    b.credit_utilization,
    b.debt_to_income,
    b.delinquencies_30d,
    avg(t.amount) over (
        partition by c.customer_id
        order by t.event_timestamp
        rows between 30 preceding and current row
    ) as avg_monthly_spend,
    sum(case when t.transaction_type = 'cash_advance' then t.amount else 0 end)
        over (partition by c.customer_id)
        / nullif(sum(t.amount) over (partition by c.customer_id), 0) as cash_advance_ratio,
    d.device_risk_score
from RISK_ANALYTICS.CORE.CUSTOMERS c
join RISK_ANALYTICS.CORE.TRANSACTIONS t
    on c.customer_id = t.customer_id
left join RISK_ANALYTICS.CORE.BUREAU_SNAPSHOTS b
    on c.customer_id = b.customer_id
left join RISK_ANALYTICS.CORE.DEVICE_EVENTS d
    on t.device_id = d.device_id;

create or replace view RISK_ANALYTICS.FEATURES.FRAUD_SIGNAL_FEATURES as
select
    t.customer_id,
    t.transaction_id,
    t.event_timestamp,
    t.amount as transaction_amount,
    m.merchant_risk_score,
    d.device_risk_score,
    count(*) over (
        partition by t.customer_id
        order by t.event_timestamp
        range between interval '1 hour' preceding and current row
    ) as login_velocity_1h,
    t.geo_velocity_km,
    datediff(day, c.account_open_date, t.event_timestamp) as account_age_days,
    extract(hour from t.event_timestamp) as hour_of_day
from RISK_ANALYTICS.CORE.TRANSACTIONS t
join RISK_ANALYTICS.CORE.CUSTOMERS c
    on t.customer_id = c.customer_id
left join RISK_ANALYTICS.CORE.MERCHANTS m
    on t.merchant_id = m.merchant_id
left join RISK_ANALYTICS.CORE.DEVICE_EVENTS d
    on t.device_id = d.device_id;
