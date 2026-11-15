# Task 006 · A credit-card recommendation

[Project overview](../README.md) · [All task results](../RESULTS.md) · [Complete reference and DeepSeek traces](../evidence/task006-traces.json)

The customer earns $95,000 a year, trades cryptocurrency and has a credit score of **540**. They require a foreign transaction fee and minimum payment percentage at or below **1.5%**, plus virtual-card management. They have no Rho-Bank+ subscription.

The reference outcome is an **EcoCard** application. Crypto-Cash Back requires a credit score of **660**. The user simulator controls the application tool and can submit as soon as it receives a recommendation.

The conversations below are condensed paraphrases of the recorded runs. Tool rows summarize the recorded operations.

## Author reference · GPT-5.2 / Terminal / trial 0

**Official reward — 1**

> **Customer**  
> My credit score is 540. Which cards meet my requirements?
>
> **Tools**  
> 48 local shell calls inspect card terms and eligibility rules.
>
> **Agent**  
> EcoCard meets your requirements.
>
> **User tool**  
> Submit an application for EcoCard.
>
> **Database check**  
> EcoCard matches the reference state.

## Successful run · DeepSeek / BM25 / trial 2

**Official reward — 1**

> **Customer**  
> I need virtual cards and low fees. My credit score is 540.
>
> **Search**  
> Fees, minimum payment, virtual cards and **credit score**. Eligibility FAQ 009 ranks ninth.
>
> **Agent**  
> Crypto-Cash Back requires 660. EcoCard fits your profile.
>
> **User tool**  
> Submit an application for EcoCard.
>
> **Database check**  
> EcoCard matches the reference state.

## Failed run · DeepSeek / BM25 / trial 0

**Official reward — 0**

> **Customer**  
> My credit score is 540. Which cards meet my requirements?
>
> **Search**  
> Fees, minimum payment and virtual cards. Results include payment document 008 and omit eligibility FAQ 009.
>
> **Agent**  
> Both EcoCard and Crypto-Cash Back work with your score.
>
> **User tool**  
> Submit an application for Crypto-Cash Back.
>
> **Database check**  
> The submitted card differs from the expected EcoCard.

Both DeepSeek trials used one search. The successful run included the customer's credit score in its query and retrieved the eligibility threshold. The runs also differ in seed and wording. This observation motivates a controlled comparison of query construction.

[Open the original DeepSeek BM25 result file](../final/baseline/runs/ds1017-4x/task_006/bm25/results.json)
