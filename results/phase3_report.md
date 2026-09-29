# Vérification Phase 3 (44 requêtes annotées, 12 requêtes de non-réponse)

## Gate reranking — recall@10 / MRR avant vs après cross-encoder

| | recall@10 | MRR |
|---|---|---|
| Hybride seul | 0.720 | 0.474 |
| Hybride + reranking | 0.686 | 0.521 |

**Gate reranking** — amélioration mesurée : **OK**

## Gate non-hallucination — set de non-réponse

12/12 cas correctement identifiés comme « je ne sais pas ».

| id | requête | answerable (doit être false) |
|---|---|---|
| n01 | how do I set up OAuth2 login in a Ruby on Rails application | False |
| n02 | what's the syntax for merging two pandas DataFrames with overlapping column names | False |
| n03 | how do I write a Kubernetes readiness probe in a deployment YAML | False |
| n04 | how does React's useEffect cleanup function work | False |
| n05 | what is the default port MySQL listens on | False |
| n06 | how do I configure rate limiting in Nginx | False |
| n07 | how do I use TensorFlow's GradientTape for custom training loops | False |
| n08 | how do goroutines communicate over channels in Go | False |
| n09 | what is the capital of France | False |
| n10 | how do I configure S3 bucket versioning with the AWS CLI | False |
| n11 | how do I write a recursive CTE in PostgreSQL | False |
| n12 | what's the difference between npm and yarn for installing packages | False |

**Gate non-hallucination** — 0/12 halluciné : **OK**
