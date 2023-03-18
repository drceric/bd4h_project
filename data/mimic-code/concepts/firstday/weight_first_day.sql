-- This query extracts weights for adult ICU patients on their first ICU day.
-- It does *not* use any information after the first ICU day, as weight is
-- sometimes used to monitor fluid balance.

-- ** Requires the echodata view, generated by concepts/echo-data.sql

with ce as
(
    SELECT
      c.icustay_id
      -- we take the avg value from roughly first day
      -- TODO: eliminate obvious outliers if there is a reasonable weight
      -- (e.g. weight of 180kg and 90kg would remove 180kg instead of taking the median)
      , AVG(VALUENUM) as Weight_Admit
    FROM `physionet-data.mimiciii_clinical.chartevents` c
    inner join `physionet-data.mimiciii_clinical.icustays` ie
        on c.icustay_id = ie.icustay_id
        and c.charttime <= (ie.intime + INTERVAL '1' DAY)
        and c.charttime > (ie.intime - INTERVAL '1' DAY) -- some fuzziness for admit time
    WHERE c.valuenum IS NOT NULL
    AND c.itemid in (762,226512) -- Admit Wt
    AND c.valuenum != 0
    -- exclude rows marked as error
    AND (c.error IS NULL OR c.error = 0)
    group by c.icustay_id
)
, dwt as
(
    SELECT
      c.icustay_id
      , AVG(VALUENUM) as Weight_Daily
    FROM `physionet-data.mimiciii_clinical.chartevents` c
    INNER JOIN `physionet-data.mimiciii_clinical.icustays` ie
        on c.icustay_id = ie.icustay_id
        and c.charttime <= (ie.intime + INTERVAL '1' DAY)
        and c.charttime > (ie.intime - INTERVAL '1' DAY) -- some fuzziness for admit time
    WHERE c.valuenum IS NOT NULL
    AND c.itemid in (763,224639) -- Daily Weight
    AND c.valuenum != 0
    -- exclude rows marked as error
    AND (c.error IS NULL OR c.error = 0)
    group by c.icustay_id
)
-- we split in-hospital/out of hospital echoes as we would like to prioritize in-hospital data
, echo_hadm as
(
    select
        ie.icustay_id
        , 0.453592*AVG(weight) as Weight_EchoInHosp
    from `physionet-data.mimiciii_notes.echo_data` ec
    inner join `physionet-data.mimiciii_clinical.icustays` ie
        on ec.hadm_id = ie.hadm_id
        and ec.charttime < (ie.intime + INTERVAL '1' DAY)
    where
            ec.HADM_ID is not null
        and ec.weight is not null
    group by ie.icustay_id
)
, echo_nohadm as
(
    select
        ie.icustay_id
        , 0.453592*AVG(weight) as Weight_EchoPreHosp
    from `physionet-data.mimiciii_notes.echo_data` ec
    inner join `physionet-data.mimiciii_clinical.icustays` ie
        on ie.subject_id = ec.subject_id
        and ie.intime < (ec.charttime + INTERVAL '1' MONTH)
        and ie.intime > ec.charttime
    where
            ec.HADM_ID is null
        and ec.weight is not null
    group by ie.icustay_id
)
select
    ie.icustay_id
    , round(cast(
    case
        when ce.icustay_id is not null
            then ce.Weight_Admit
        when dwt.icustay_id is not null
            then dwt.Weight_Daily
        when eh.icustay_id is not null
            then eh.Weight_EchoInHosp
        when enh.icustay_id is not null
            then enh.Weight_EchoPreHosp
        else null end
        as numeric), 2)
    as weight

    -- components
    , ce.weight_admit
    , dwt.weight_daily
    , eh.weight_echoinhosp
    , enh.weight_echoprehosp

FROM `physionet-data.mimiciii_clinical.icustays` ie

-- filter to only adults
inner join `physionet-data.mimiciii_clinical.patients` pat
    on ie.subject_id = pat.subject_id
    and ie.intime > (pat.dob + INTERVAL '1' YEAR)

-- admission weight
left join ce
    on ie.icustay_id = ce.icustay_id

-- daily weights
left join dwt
    on ie.icustay_id = dwt.icustay_id

-- in-hospital echo weight
left join echo_hadm eh
    on ie.icustay_id = eh.icustay_id

-- pre-hospitalization echo weights
left join echo_nohadm enh
    on ie.icustay_id = enh.icustay_id
order by ie.icustay_id;
