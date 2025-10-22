## 📊 Analysis and Key Findings

### 1. Data Summary
* **Total Reviews Analyzed:** **9,093** (all collected in the year 2025).
* **Data Columns:** `reviewId`, `userName`, `score`, `content`, `at` (timestamp), `thumbsUpCount`, `reviewCreatedVersion`, `content_clean`, `intent`, `sentiment`.

### 2. Star Ratings Distribution
The distribution of 1-5 star ratings is heavily skewed toward the extremes, a common pattern in online reviews.
* **1-Star** and **5-Star** reviews are the most frequent.



### 3. Rule-Based Intent Classification
A simple keyword-matching approach was used to categorize the user's main intent.

| Intent | Count | Key Keywords |
| :--- | :--- | :--- |
| **neutral_or_other** | 4432 | (Fallback for no match) |
| **criticism** | 1839 | "terrible", "worst", "fake", "scam" |
| **positive** | 1589 | "love", "great", "best", "met someone" |
| **suggestion** | 431 | "should", "please add", "feature" |
| **bug** | 404 | "crash", "error", "not working", "stuck" |
| **question** | 344 | "how do i", "why", "can you", "?" |
| **spam** | 54 | "link", "visit", "earn money" |

**Observation:** While nearly half the reviews are unclassified by the basic rules (`neutral_or_other`), **Criticism** and **Positive** feedback are the two dominant themes among classified reviews, followed by **Suggestions** and **Bug** reports.

---

### 4. Sentiment Analysis with VADER
VADER (Valence Aware Dictionary and sEntiment Reasoner) was used to classify each review's sentiment based on its `compound` score:

* **Positive**: $\ge 0.05$
* **Neutral**: $(-0.05, 0.05)$
* **Negative**: $\le -0.05$

#### Sentiment Distribution
The sentiment is nearly evenly split between positive and negative:

| Sentiment | Proportion |
| :--- | :--- |
| **Positive** | 43% |
| **Negative** | 43% |
| **Neutral** | 14% |


#### Sentiment Trend Over Time (2025)
The sentiment trend suggests a relatively **stable proportion of positive, negative, and neutral** reviews throughout the year. There are no major spikes indicating sudden user satisfaction or widespread issues.
