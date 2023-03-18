-- This code extracts structured data from echocardiographies
-- You can join it to the text notes using ROW_ID
-- Just note that ROW_ID will differ across versions of MIMIC-III.

select ROW_ID
  , subject_id, hadm_id
  , chartdate

  -- charttime is always null for echoes..
  -- however, the time is available in the echo text, e.g.:
  -- , substring(ne.text, 'Date/Time: [\[\]0-9*-]+ at ([0-9:]+)') as TIMESTAMP
  -- we can therefore impute it and re-create charttime
  , TO_TIMESTAMP(
    CONCAT(
        FORMAT(chartdate, 'YYYY-MM-DD'),
        REGEXP_REPLACE(ne.text, '^.*Date/Time: .+? at ([0-9]+:[0-9]{2}).*$', ' \1:00')
    ),
    'YYYY-MM-DD HH24:MI:SS'
) AS charttime
  -- explanation of below substring:
  --  'Indication: ' - matched verbatim
  --  (.*?) - match any character
  --  \n - the end of the line
  -- substring only returns the item in ()s
  -- note: the '?' makes it non-greedy. if you exclude it, it matches until it reaches the *last* \n

, SUBSTRING(ne.text, 'Indication: (.*?)\n', 1, 'g') AS Indication

-- sometimes numeric values contain de-id text, e.g. [** Numeric Identifier **]
-- this removes that text
, CAST(SUBSTRING(ne.text, 'Height: \\x28in\\x29 ([0-9]+)', 1, 'g') AS numeric) AS Height
, CAST(SUBSTRING(ne.text, 'Weight \\x28lb\\x29: ([0-9]+)\n', 1, 'g') AS numeric) AS Weight
, CAST(SUBSTRING(ne.text, 'BSA \\x28m2\\x29: ([0-9]+) m2\n', 1, 'g') AS numeric) AS BSA -- ends in 'm2'
, SUBSTRING(ne.text, 'BP \\x28mm Hg\\x29: (.+)\n', 1, 'g') AS BP -- Sys/Dias
, CAST(SUBSTRING(ne.text, 'BP \\x28mm Hg\\x29: ([0-9]+)/[0-9]+?\n', 1, 'g') AS numeric) AS BPSys -- first part of fraction
, CAST(SUBSTRING(ne.text, 'BP \\x28mm Hg\\x29: [0-9]+/([0-9]+?)\n', 1, 'g') AS numeric) AS BPDias -- second part of fraction
, CAST(SUBSTRING(ne.text, 'HR \\x28bpm\\x29: ([0-9]+?)\n', 1, 'g') AS numeric) AS HR

, SUBSTRING(ne.text, 'Status: (.*?)\n', 1, 'g') AS Status
, SUBSTRING(ne.text, 'Test: (.*?)\n', 1, 'g') AS Test
, SUBSTRING(ne.text, 'Doppler: (.*?)\n', 1, 'g') AS Doppler
, SUBSTRING(ne.text, 'Contrast: (.*?)\n', 1, 'g') AS Contrast
, SUBSTRING(ne.text, 'Technical Quality: (.*?)\n', 1, 'g') AS TechnicalQuality

FROM `physionet-data.mimiciii_notes.noteevents` ne
where category = 'Echo';
