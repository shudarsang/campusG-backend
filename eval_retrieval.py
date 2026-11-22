"""Answer-level retrieval evaluation. No Gemini calls. Run from campusG-backend/.

eval_zia.py checks that the right *file/section* comes back for
questions worded much like the FAQ. This checks something stricter
and closer to what users feel: for casually-worded questions, does
any of the top-k chunks actually *contain the answer*?

Each case is (question, [regex, ...]); a chunk counts as relevant if
any regex matches its text (case-insensitive).

    python eval_retrieval.py            # summary + misses
    python eval_retrieval.py -v         # every question with ranks
"""
import re
import sys
import time

sys.path.insert(0, ".")

from app.rag.retriever import retriever  # noqa: E402

CASES = [
    # Facilities / campus
    ("is there hostel facility", [r"\b540\b"]),
    ("how many students can stay in the hostel", [r"\b540\b"]),
    ("who gets priority for hostel seats", [r"first-generation learners, students with disab"]),
    ("how do i get a hostel room", [r"first-come"]),
    ("library timing", [r"9:00 a\.m\. to 4:00 p\.m"]),
    ("what time does college start and end", [r"2:15"]),
    ("which sports can i play here", [r"kabaddi|throwball|basketball, volleyball"]),
    ("indoor stadium cost", [r"5 crores"]),
    ("is there a gym", [r"gym facility available within the hostel|within the hostel"]),
    ("is there a doctor or medical help on campus", [r"physiotherapy"]),
    ("does the library have online journals or databases", [r"EBSCO|INFLIBNET"]),
    ("braille books for blind students", [r"braille"]),
    ("what is ECRIC", [r"Research, Innovation and Creativity"]),
    # Contact / general
    ("college phone number", [r"28279189"]),
    ("principal email id", [r"principal@ethirajcollege"]),
    ("where is the college", [r"Ethiraj Salai"]),
    ("fax number", [r"28282014"]),
    ("who started this college", [r"V\.\s?L\.\s?Ethiraj"]),
    ("which year was it established", [r"\b1948\b"]),
    ("college motto", [r"not to yield"]),
    ("how many students study here", [r"7,000|7,700"]),
    ("is it affiliated to any university", [r"University of Madras"]),
    ("naac grade", [r"A\+"]),
    ("nirf rank", [r"\b79\b|\b94\b"]),
    ("college with potential for excellence", [r"\b2004\b"]),
    # Placements
    ("which companies come for placement", [r"Infosys|Wipro|Accenture"]),
    ("who is the placement officer", [r"Gomathy"]),
    ("entrepreneurship cell event", [r"IDEA CONQUEST"]),
    # Admissions
    ("do i need to write tancet for mba", [r"TANCET"]),
    ("mca eligibility", [r"Mathematics"]),
    ("is there an entrance test for bcom", [r"no entrance|purely on (?:10\+2 )?merit|merit"]),
    ("documents needed for applying", [r"Community Certificate"]),
    ("photo size for the application form", [r"100\s?kb"]),
    ("can i apply to more than one course", [r"any number of courses"]),
    ("can i change my application after submitting", [r"cannot be edited"]),
    ("last date to apply", [r"31 July|6 June"]),
    ("how to pay the fees online", [r"Pay Online|payment link|online payment"]),
    ("scholarship from alumni", [r"Alumnae Association"]),
    # Academics
    ("bcom hons eligibility", [r"Accountancy"]),
    ("how long is mca", [r"2[- ]years?|two years"]),
    ("when did the mca course start", [r"\b1999\b"]),
    ("hod of computer science", [r"Jane Nithya"]),
    ("how many ug courses are there", [r"UG Programmes:\*\* 19|\b19\b.*UG|UG.*\b19\b"]),
    ("bsc microbiology duration", [r"Microbiology[\s\S]*3 Years"]),
    ("bcom bank management department", [r"Bank Management[\s\S]*Commerce"]),
    # One-word and generic questions, where near-duplicate FAQ
    # templates ("Is X available for <programme> students?") crowd
    # out the chunks that hold the actual facts.
    ("sports", [r"kabaddi|basketball, volleyball|Indoor Stadium"]),
    ("Is hostel accommodation available?", [r"\b540\b|first-come"]),
    ("library", [r"4:00 p\.m|EBSCO|braille"]),
    # Typos - users type fast, and the voice transcript is not perfect.
    ("adress of colege", [r"Ethiraj Salai"]),
    ("bcom elegibility", [r"Accountancy"]),
    ("hostal capacity", [r"\b540\b"]),
    ("libary timings", [r"4:00 p\.m"]),
    ("plcement companies", [r"Infosys|Wipro|Accenture"]),
    ("is tancet compulsary for mca", [r"TANCET"]),
    ("principle of the college", [r"Principal"]),
]

# Boilerplate FAQ answers that say nothing specific.
BOILERPLATE = re.compile(
    r"generally available college-wide|should be confirmed directly", re.I
)

K = 6


def first_relevant_rank(docs, patterns):
    for rank, doc in enumerate(docs, 1):
        if any(re.search(p, doc.page_content, re.I) for p in patterns):
            return rank
    return None


def main(verbose=False):
    hit1 = hit3 = hitk = 0
    rr = 0.0
    misses = []
    latencies = []
    boilerplate = 0
    for question, patterns in CASES:
        t = time.perf_counter()
        docs = retriever.retrieve(question, k=K)
        latencies.append((time.perf_counter() - t) * 1000)
        boilerplate += sum(bool(BOILERPLATE.search(d.page_content)) for d in docs)
        rank = first_relevant_rank(docs, patterns)
        hit1 += rank == 1
        hit3 += bool(rank and rank <= 3)
        hitk += bool(rank)
        rr += 1 / rank if rank else 0
        if verbose:
            print(f"{str(rank or '-'):>2}  {question}")
        if not rank:
            misses.append((question, docs))

    n = len(CASES)
    print(f"\nquestions={n}  hit@1={hit1/n:.3f}  hit@3={hit3/n:.3f}  "
          f"hit@{K}={hitk/n:.3f}  mrr={rr/n:.3f}  "
          f"latency_ms_median={sorted(latencies)[n // 2]:.1f}  "
          f"boilerplate_chunks_per_answer={boilerplate / n:.2f}")
    print(f"\nMisses ({len(misses)}):")
    for question, docs in misses:
        print(f"  - {question}")
        for d in docs[:3]:
            print(f"      {d.metadata.get('section', '')[:90]}")


if __name__ == "__main__":
    main(verbose="-v" in sys.argv)
