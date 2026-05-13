import os
import time
import json
from dotenv import load_dotenv
load_dotenv()

from groq import Groq
from sentence_transformers import SentenceTransformer, util
from retriever import retrieve

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
similarity_model = SentenceTransformer("intfloat/multilingual-e5-base", device="cuda")

MODELS = [
    "meta-llama/llama-4-scout-17b-16e-instruct",
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
]

TEST_SET = [

    # ── Basic policy ───────────────────────────────────────────
    {
        "question": "My phone screen cracked during delivery, can I return it?",
        "ground_truth": "Yes, physically damaged products are eligible for return or replacement within 7 or 10 days of delivery. Keep the original packaging and accessories intact.",
        "category": "basic_policy"
    },
    {
        "question": "I got the wrong color of shoes. Can I exchange them?",
        "ground_truth": "Yes, wrong items delivered are eligible for free replacement within 7 or 10 days. You may also return for a full refund if the seller has the correct item in stock.",
        "category": "basic_policy"
    },
    {
        "question": "What items do I need to keep for a successful return pickup?",
        "ground_truth": "Keep the item in original condition with brand outer box, MRP tags, user manual, warranty cards, and all original accessories in manufacturer packaging.",
        "category": "basic_policy"
    },

    # ── Multi-hop reasoning ────────────────────────────────────
    {
        "question": "I ordered a laptop from abroad and it arrived defective. Can I get a refund and how long do I have?",
        "ground_truth": "International customers are not eligible for returns, but can claim refunds for defective items. You must contact customer service within 5 business days from the delivery date.",
        "category": "multi_hop"
    },
    {
        "question": "The seller I bought from does not have my item in stock for replacement and the product is damaged. What happens now?",
        "ground_truth": "If the seller does not have the exact same product in stock for replacement, Amazon will issue a full refund instead of a replacement for the damaged item.",
        "category": "multi_hop"
    },
    {
        "question": "I want to return my refrigerator but it is making a strange noise. Do I return it or call a technician?",
        "ground_truth": "For appliances like refrigerators, Amazon may facilitate a technician visit to your location for troubleshooting. Final resolution such as repair, refund, or replacement will be based on the technician's evaluation report.",
        "category": "multi_hop"
    },

    # ── Ambiguous / edge cases ─────────────────────────────────
    {
        "question": "I bought a TV during a sale 3 weeks ago and I just don't like it anymore. Can I return it?",
        "ground_truth": "No. Buyer's remorse such as not liking a product is not a valid reason for return. Returns are only accepted for damaged, defective, or wrong items within the return window.",
        "category": "edge_case"
    },
    {
        "question": "I opened my laptop and used it for a week, but now found it has a defect. What are my options?",
        "ground_truth": "For defective laptops, you should reach out to the brand service centre. Amazon may also facilitate a technician visit for troubleshooting. Resolution such as repair or replacement depends on the technician's report.",
        "category": "edge_case"
    },
    {
        "question": "The product page said non-returnable but my item came damaged. Am I stuck with it?",
        "ground_truth": "No. Even non-returnable products are eligible for full refund or replacement if they arrive damaged, defective, or different from what was described. Contact Amazon within 5 days of delivery.",
        "category": "edge_case"
    },
    {
        "question": "I shared my OTP with the delivery agent after checking the product. Now I see a missing accessory. Can I still return it?",
        "ground_truth": "No. If OTP was shared after verifying the product through the Inspection Service, you cannot return the product claiming it is wrong, damaged, or missing accessories.",
        "category": "edge_case"
    },

    # ── Amazon Prime questions ─────────────────────────────────
    {
        "question": "What benefits do I get with Amazon Prime membership?",
        "ground_truth": "Amazon Prime offers benefits including free and fast delivery, access to Prime Video, Prime Music, early access to deals, and exclusive member offers.",
        "category": "prime"
    },
    {
        "question": "Can I cancel my Amazon Prime membership and get a refund?",
        "ground_truth": "Yes, you can cancel your Amazon Prime membership. Refund eligibility depends on whether any Prime benefits have been used since the last charge. Unused memberships may be eligible for a full refund.",
        "category": "prime"
    },
    {
        "question": "How much does Amazon Prime cost and how often am I charged?",
        "ground_truth": "Amazon Prime membership has a monthly or annual fee. The exact amount and billing frequency depends on the plan selected at the time of subscription.",
        "category": "prime"
    },
    {
        "question": "I was charged for Prime but I never signed up for it. What do I do?",
        "ground_truth": "You should contact Amazon customer service immediately to dispute the charge. Amazon may investigate and issue a refund if the subscription was not intentionally activated by you.",
        "category": "prime"
    },

    # ── Promotions and offers ──────────────────────────────────
    {
        "question": "I used a coupon on my order. If I return it, will I get the full price back or just what I paid?",
        "ground_truth": "Refunds are typically issued for the amount actually paid after discounts and coupons. The coupon or promotional discount is generally not refunded as separate credit.",
        "category": "promotions"
    },
    {
        "question": "There is a cashback offer on my credit card for Amazon purchases. Will I lose it if I return the item?",
        "ground_truth": "Cashback and bank offers are managed by the respective bank or card issuer. Returning an item may result in reversal of the cashback as per the bank's terms. Amazon's refund covers only the amount paid to Amazon.",
        "category": "promotions"
    },
    {
        "question": "Can I use multiple promo codes on a single order?",
        "ground_truth": "Generally only one promotional code or offer can be applied per order on Amazon. The specific terms depend on the promotion and are mentioned in the offer details.",
        "category": "promotions"
    },

    # ── Tracking and delivery ──────────────────────────────────
    {
        "question": "My order shows delivered but I never received it. What should I do?",
        "ground_truth": "If your order shows as delivered but you have not received it, contact Amazon customer service immediately. You should report it within 30 days of the estimated delivery date to be eligible for a refund or replacement.",
        "category": "delivery"
    },
    {
        "question": "How do I track my order after it has been shipped?",
        "ground_truth": "You can track your order by visiting the Your Orders section in your Amazon account. The tracking details including carrier and tracking number will be available once the order is shipped.",
        "category": "delivery"
    },
    {
        "question": "My order is stuck in transit for 10 days. What can I do?",
        "ground_truth": "If your order has been in transit for an unusually long time, contact Amazon customer service with your order details. They can investigate with the carrier and arrange a refund or replacement if the item is lost.",
        "category": "delivery"
    },

    # ── Hindi multilingual ─────────────────────────────────────
    {
        "question": "मेरा प्रोडक्ट damaged आया है, मैं इसे कैसे return करूं?",
        "ground_truth": "Damaged products can be returned within 7 or 10 days of delivery. Go to Your Orders, select the item, and choose Return or Replace. Keep original packaging and accessories ready for pickup.",
        "category": "hindi"
    },
    {
        "question": "Amazon Prime का membership fee कितना है और क्या मैं इसे cancel कर सकता हूं?",
        "ground_truth": "Amazon Prime has a monthly or annual fee. You can cancel your membership anytime. Refund depends on whether Prime benefits have been used after the last billing.",
        "category": "hindi"
    },
    {
        "question": "मुझे refund कब तक मिलेगा अगर मैंने product return किया?",
        "ground_truth": "Refunds are typically processed within 3 to 5 business days after the returned item is picked up and verified by Amazon.",
        "category": "hindi"
    },

    # ── Account and policy abuse ───────────────────────────────
    {
        "question": "I have returned more than 10 orders this month. Will Amazon block my account?",
        "ground_truth": "Yes. Amazon reserves the right to warn, suspend, block, or terminate accounts that are found to misuse the return policy through excessive returns or order cancellations.",
        "category": "account"
    },
    {
        "question": "Can I return a product I bought as a gift if the recipient does not like it?",
        "ground_truth": "Returns are only accepted for damaged, defective, or wrong items within the return window. Change of mind or personal preference such as a gift recipient not liking the item is generally not a valid return reason unless the product qualifies under the returnable policy.",
        "category": "account"
    },
]


# ── Metrics ────────────────────────────────────────────────────
def answer_similarity(answer: str, ground_truth: str) -> float:
    emb1 = similarity_model.encode(f"query: {answer}", convert_to_tensor=True)
    emb2 = similarity_model.encode(f"query: {ground_truth}", convert_to_tensor=True)
    return round(float(util.cos_sim(emb1, emb2)), 4)


def context_relevance(question: str, contexts: list[str]) -> float:
    q_emb = similarity_model.encode(f"query: {question}", convert_to_tensor=True)
    scores = [
        float(util.cos_sim(q_emb, similarity_model.encode(f"passage: {c}", convert_to_tensor=True)))
        for c in contexts
    ]
    return round(sum(scores) / len(scores), 4)




# ── Generator ──────────────────────────────────────────────────
def generate_answer(query: str, model_name: str, top_k: int = 5) -> dict:
    results = retrieve(query, top_k=top_k)
    context_texts = [r["text"] for r in results]
    context_block = "\n\n".join([f"[{i+1}] {t}" for i, t in enumerate(context_texts)])

    start = time.time()
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "system",
                "content": """You are a helpful Amazon India customer service agent.
Answer the user's question using ONLY the context provided.
If the context doesn't contain enough information, say so honestly.
Be concise, direct, and empathetic.""",
            },
            {
                "role": "user",
                "content": f"Context:\n{context_block}\n\nQuestion: {query}",
            },
        ],
    )
    latency = round(time.time() - start, 3)

    return {
        "answer":   response.choices[0].message.content,
        "contexts": context_texts,
        "latency":  latency,
    }


# ── Run Evaluation ─────────────────────────────────────────────
def run_evaluation(model_name: str) -> dict:
    label = model_name.split("/")[-1]
    print(f"\n{'='*65}")
    print(f"  Evaluating: {label}")
    print(f"{'='*65}")

    results_per_query = []
    category_scores = {}

    for i, item in enumerate(TEST_SET):
        print(f"  [{i+1}/{len(TEST_SET)}] [{item['category']}] {item['question'][:50]}...")
        result   = generate_answer(item["question"], model_name)
        answer   = result["answer"]
        contexts = result["contexts"]
        latency  = result["latency"]

        sim    = answer_similarity(answer, item["ground_truth"])
        ctx_r  = context_relevance(item["question"], contexts)

        print(f"    sim={sim} | ctx={ctx_r} | {latency}s")

        row = {
            "question":          item["question"],
            "category":          item["category"],
            "answer":            answer,
            "ground_truth":      item["ground_truth"],
            "answer_similarity": sim,
            "context_relevance": ctx_r,
            "latency":           latency,
        }
        results_per_query.append(row)

        # track per category
        cat = item["category"]
        if cat not in category_scores:
            category_scores[cat] = []
        category_scores[cat].append(sim)

    avg = lambda key: round(sum(r[key] for r in results_per_query) / len(results_per_query), 4)

    cat_avg = {
        cat: round(sum(scores) / len(scores), 4)
        for cat, scores in category_scores.items()
    }

    summary = {
        "model":                  model_name,
        "label":                  label,
        "avg_answer_similarity":  avg("answer_similarity"),
        "avg_context_relevance":  avg("context_relevance"),
        "avg_latency":            avg("latency"),
        "category_scores":        cat_avg,
        "per_query":              results_per_query,
    }

    print(f"\n── Summary: {label} ──")
    print(f"  Answer Similarity  : {summary['avg_answer_similarity']}")
    print(f"  Context Relevance  : {summary['avg_context_relevance']}")
    print(f"  Avg Latency        : {summary['avg_latency']}s")
    print(f"\n  Per Category (answer similarity):")
    for cat, score in cat_avg.items():
        print(f"    {cat:<20} : {score}")

    return summary


# ── Main ───────────────────────────────────────────────────────
if __name__ == "__main__":
    all_results = []

    for model in MODELS:
        summary = run_evaluation(model)
        all_results.append(summary)

    with open("eval_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print("\n" + "="*75)
    print("  FINAL COMPARISON")
    print("="*75)
    print(f"  {'Model':<38} {'Sim':>6} {'CtxRel':>8} {'Latency':>9}")
    print(f"  {'-'*65}")
    for r in all_results:
        print(
            f"  {r['label']:<38}"
            f"  {r['avg_answer_similarity']:>6}"
            f"  {r['avg_context_relevance']:>8}"
            f"  {str(r['avg_latency'])+'s':>9}"
        )

    print("\n  Per Category breakdown:")
    cats = list(all_results[0]["category_scores"].keys())
    print(f"  {'Category':<22}", end="")
    for r in all_results:
        print(f"  {r['label'][:18]:>18}", end="")
    print()
    print(f"  {'-'*80}")
    for cat in cats:
        print(f"  {cat:<22}", end="")
        for r in all_results:
            print(f"  {r['category_scores'].get(cat, 0):>18}", end="")
        print()

    print("\n✅ Results saved → eval_results.json") 