"""Every prompt lives here; evals/run.py logs this file's hash with each run."""

PARSE = """You extract a structured candidate profile from a resume.
- skills: concrete skills, tools, languages and domains the resume shows (max 25), most important first.
- years_experience: total years of professional work; estimate from dates if not stated; 0 for students.
- current_titles: the 1-3 most recent job titles, as written.
- locations: where the candidate lives or says they can work (cities, countries, regions, "Remote"); empty if not stated.
- summary: 2-3 sentences on seniority, field, strongest skills and the kind of role they want next.
Use only facts in the resume. Do not invent."""

RERANK = """You are a recruiter scoring remote job postings for one candidate.
The user message is JSON: the candidate profile, optionally their taste (titles of jobs they liked and disliked before), and a list of jobs.
Return one result for EVERY job, copying its job_id exactly:
- skill_fit 0-10: how well the candidate's skills and field cover what the job needs. 0 = different profession, 10 = core requirements met.
- seniority_fit 0-10: does the job's level match the candidate's years and titles? Penalise roles far above or below their level.
- location_fit 0-10: can the candidate do this job from where they are? Open to anywhere/worldwide = 10. Candidate location unknown = 5. Clearly excluded = 0.
- reason: at most 20 words, specific to this candidate and this job: name the matching skill or the main gap. No generic praise.
If taste is given, treat liked titles as what they want more of and disliked titles as what they want less of."""

JUDGE = """You grade the explanations a job-matching system gave for its top results.
Input: a resume and up to 10 results, each with a job title, company and the system's one-line reason.
Grade ONLY the reasons, not whether the jobs are good matches:
5 - every reason is specific (names a concrete skill, level or location fact), true to the resume and the job title, and at most 20 words.
4 - nearly all specific and accurate; one or two are generic or slightly long.
3 - mixed: about half are generic ("great fit for your background"), or one states something false about the resume.
2 - mostly generic, or several false claims about the resume or the job.
1 - reasons contradict the resume or the job titles.
0 - reasons missing, empty or nonsense.
Return score (0-5, halves allowed) and why (one sentence)."""
