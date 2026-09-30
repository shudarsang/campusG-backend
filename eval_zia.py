"""Offline evaluation of Zia's router (keyword stage) and hybrid retriever.
No Gemini calls are made. Run from campusG-backend/."""
import sys, time, statistics, json
sys.path.insert(0, ".")
from app.agents.router_agent import router
from app.rag.retriever import retriever

# (question, expected agent, [(source file, section substring or "")])
CASES = [
 ("What is the eligibility for MCA?", "admission", [("admissions.md",""),("entrance_exams.md",""),("courses.md","MCA"),("faq.md","Admissions"),("faq.md","Courses")]),
 ("What documents are required for admission?", "admission", [("admissions.md","Documents"),("faq.md","Admissions")]),
 ("When do admissions close?", "admission", [("admissions.md","Dates"),("faq.md","Admissions"),("news_events.md","")]),
 ("What is the fee structure for UG courses?", "admission", [("fee_structure.md",""),("faq.md","Fees")]),
 ("What scholarships are available?", "admission", [("scholarships.md",""),("admissions.md","Scholarships"),("faq.md","Scholarships")]),
 ("Which entrance exams are accepted for MBA?", "admission", [("entrance_exams.md",""),("faq.md","Entrance")]),
 ("How do I apply online?", "admission", [("admissions.md",""),("faq.md","Admissions")]),
 ("What is the reservation policy?", "admission", [("admissions.md","Reservation"),("faq.md","Admissions")]),
 ("Is there lateral entry admission?", "admission", [("admissions.md","Lateral"),("faq.md","Admissions")]),
 ("What is the selection procedure for PG programmes?", "admission", [("admissions.md","Selection"),("entrance_exams.md",""),("faq.md","Admissions")]),
 ("What courses are offered?", "academic", [("courses.md",""),("faq.md","Courses")]),
 ("List the PG programmes", "academic", [("courses.md","Postgraduate"),("faq.md","Courses")]),
 ("Does the college offer Ph.D programmes?", "academic", [("courses.md","Research"),("research.md",""),("faq.md","Research"),("faq.md","Courses")]),
 ("Which departments are there?", "academic", [("departments.md",""),("faq.md","Departments")]),
 ("Who is the head of the Computer Science department?", "academic", [("faculty.md",""),("departments.md",""),("faq.md","Departments")]),
 ("Tell me about the B.Sc. Microbiology course", "academic", [("courses.md",""),("departments.md",""),("faq.md","Courses"),("faq.md","Departments")]),
 ("What is the examination system?", "academic", [("faq.md","Academics"),("faq.md","")]),
 ("What research centres does the college have?", "academic", [("research.md",""),("facilities.md","Research"),("faq.md","Research")]),
 ("What are the placement statistics?", "campus", [("placements.md",""),("faq.md","Placements"),("facilities.md","Placement")]),
 ("Which companies visit for placements?", "campus", [("placements.md",""),("faq.md","Placements")]),
 ("What are the library timings?", "campus", [("facilities.md","Library"),("faq.md","Facilities")]),
 ("Is hostel accommodation available?", "campus", [("facilities.md","Hostel"),("faq.md","Facilities"),("faq.md","Campus")]),
 ("What sports facilities are available?", "campus", [("facilities.md","Sports"),("campus_life.md","Sports"),("faq.md","Facilities"),("faq.md","Campus")]),
 ("Is Wi-Fi available on campus?", "campus", [("facilities.md","WiFi"),("faq.md","Facilities")]),
 ("Does the college provide transport?", "campus", [("facilities.md","Transport"),("faq.md","Facilities")]),
 ("What clubs can students join?", "campus", [("campus_life.md",""),("faq.md","Campus")]),
 ("Is there an NCC unit?", "campus", [("facilities.md","NCC"),("campus_life.md","NCC"),("faq.md","")]),
 ("Is there a canteen in the college?", "campus", [("facilities.md","Cafeteria"),("faq.md","Facilities")]),
 ("Are medical facilities available on campus?", "campus", [("facilities.md","Medical"),("faq.md","Facilities")]),
 ("Is there a gym?", "campus", [("facilities.md","Gym"),("faq.md","Facilities")]),
 ("When was the college founded?", "general", [("about_college.md",""),("faq.md","About")]),
 ("Who founded Ethiraj College?", "general", [("about_college.md","Founder"),("mission_vision.md","Founder"),("faq.md","About")]),
 ("What is the vision of the college?", "general", [("mission_vision.md","Vision"),("faq.md","Mission")]),
 ("What is the NIRF ranking?", "general", [("rankings.md",""),("about_college.md","Rankings"),("faq.md","Rankings")]),
 ("What is the NAAC grade?", "general", [("rankings.md",""),("about_college.md","Accreditation"),("faq.md","Rankings")]),
 ("Who is the principal?", "general", [("administration.md","Principal"),("faq.md","Administration")]),
 ("What is the college address?", "general", [("contact.md","Address"),("about_college.md","Location"),("faq.md","Contact")]),
 ("What is the contact phone number?", "general", [("contact.md","Phone"),("faq.md","Contact")]),
 ("What is the college motto?", "general", [("about_college.md","Motto"),("mission_vision.md","Motto"),("faq.md","About")]),
 ("What are the latest news and events?", "general", [("news_events.md",""),("campus_life.md","Events")]),
]

def relevant(doc, expected):
    src = doc.metadata.get("source", ""); sec = doc.metadata.get("section", "")
    return any(src == s and (sub.lower() in sec.lower()) for s, sub in expected)

K = 6
route_correct = route_decided = route_decided_correct = 0
hit1 = hit3 = hitk = 0; rr = []; lat = []; per_agent = {}
rows = []
for q, agent, exp in CASES:
    kw = router._keyword_route(q)
    decided = kw is not None
    route_decided += decided
    final = kw or "general"   # without the LLM fallback, general is the default
    route_decided_correct += decided and kw == agent
    route_correct += final == agent
    t = time.perf_counter(); docs = retriever.retrieve(q, k=K); lat.append((time.perf_counter()-t)*1000)
    ranks = [i for i, d in enumerate(docs, 1) if relevant(d, exp)]
    first = ranks[0] if ranks else None
    hit1 += first == 1; hit3 += bool(first and first <= 3); hitk += bool(first)
    rr.append(1/first if first else 0)
    a = per_agent.setdefault(agent, [0, 0, 0]); a[0] += 1; a[1] += bool(first); a[2] += (final == agent)
    rows.append((q, agent, kw or "-> LLM fallback", first))

n = len(CASES)
print(json.dumps({
 "questions": n, "chunks_indexed": len(retriever._documents),
 "router_keyword_decided": route_decided, "router_keyword_decided_correct": route_decided_correct,
 "router_accuracy_keyword_plus_general_default": route_correct / n,
 "hit@1": hit1 / n, "hit@3": hit3 / n, f"hit@{K}": hitk / n, "mrr": sum(rr) / n,
 "latency_ms_mean": statistics.mean(lat[1:]), "latency_ms_median": statistics.median(lat[1:]),
 "latency_ms_p95": sorted(lat[1:])[int(0.95 * (n - 1)) - 1], "latency_first_ms": lat[0],
 "per_agent": per_agent}, indent=1))
for r in rows: print(r)
