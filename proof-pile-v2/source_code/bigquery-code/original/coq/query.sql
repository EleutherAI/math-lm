SELECT
f.repo_name,
f.path,
c.copies,
c.size,
c.content,
l.license
FROM
`bigquery-public-data.github_repos.files` AS f
JOIN
`bigquery-public-data.github_repos.contents` AS c
ON
f.id = c.id
JOIN
`bigquery-public-data.github_repos.licenses` AS l
ON
f.repo_name = l.repo_name
WHERE
NOT c.binary
AND ((f.path LIKE '%.v')
AND (c.size BETWEEN 1024
AND 1048575))