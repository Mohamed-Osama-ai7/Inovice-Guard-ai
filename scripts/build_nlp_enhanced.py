from __future__ import annotations

import json
import random
from itertools import product
from pathlib import Path
from typing import Iterable, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"

DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(exist_ok=True)

LABEL_ORDER = ["LOW_RISK", "MEDIUM_RISK", "HIGH_RISK"]
SEED = 42


def sample_product(prefixes: List[str], actions: List[str], suffixes: List[str], limit: int) -> List[str]:
    options = []
    for prefix, action, suffix in product(prefixes, actions, suffixes):
        text = " ".join([prefix.strip(), action.strip(), suffix.strip()])
        if text not in options:
            options.append(text)
    rng = random.Random(SEED)
    rng.shuffle(options)
    return options[:limit]


def build_nlp_candidate_sets() -> List[dict]:
    families = []

    low_specs = [
        {
            "group_id": "LOW_01_COMPLETE",
            "prefixes": [
                "Payment has already been completed.",
                "The transfer was successfully sent.",
                "The invoice is fully settled.",
                "Funds were released this morning.",
                "The account has already been credited.",
                "The payment was completed on schedule.",
                "Settlement is confirmed and the remittance has been posted.",
                "The outstanding balance has been cleared.",
            ],
            "actions": [
                "The invoice is now closed.",
                "No further action is required.",
                "The customer has already satisfied the obligation.",
                "There are no outstanding payment issues.",
                "The remittance advice has been delivered.",
                "The transaction is fully reconciled.",
                "The payment status is confirmed.",
                "The agreement is satisfied in full.",
            ],
            "suffixes": [
                "No payment disruption is expected.",
                "The due date is effectively resolved.",
                "The invoice should not require any additional follow-up.",
                "This matter is already settled from a payment perspective.",
                "There is no current blocker on the account.",
                "The funds are already reflected in the account.",
                "The payment activity has already been completed.",
                "The customer has completed the requested transfer.",
            ],
        },
        {
            "group_id": "LOW_02_APPROVED",
            "prefixes": [
                "Finance has approved the invoice for payment.",
                "The payment request has been cleared internally.",
                "Treasury has confirmed the invoice is ready to pay.",
                "The invoice has been approved and positioned for release.",
                "The payment is authorized and waiting for the transfer step.",
                "All review steps have been completed and the invoice is approved.",
                "Internal approval has already been secured.",
                "The customer has confirmed the invoice and its payment is approved.",
            ],
            "actions": [
                "The transfer will be processed as planned.",
                "The payment is on track for settlement.",
                "The bank transfer is expected to go out on time.",
                "The funds are scheduled to move according to the agreement.",
                "The release is already prepared in the payment queue.",
                "The transfer should happen without additional delay.",
                "Processing is underway and proceeds according to schedule.",
                "The payment has been routed for execution.",
            ],
            "suffixes": [
                "This should meet the original due date.",
                "There is no sign of a payment problem.",
                "The customer intends to settle without any interruption.",
                "The account is expected to reflect the transfer soon.",
                "The payment timing is considered normal.",
                "The outstanding amount is expected to clear on schedule.",
                "The invoice is aligned with the existing payment plan.",
                "There is no indication of refusal or cancellation.",
            ],
        },
        {
            "group_id": "LOW_03_SCHEDULED",
            "prefixes": [
                "The payment is scheduled for tomorrow.",
                "Funds are planned to be released this week.",
                "The customer expects to transfer the amount shortly.",
                "The invoice is queued for payment as agreed.",
                "The payment is expected to arrive on the committed date.",
                "The transfer has been arranged and should go through promptly.",
                "Everything is in place for the payment to be sent soon.",
                "The custodian has confirmed the payment release window.",
            ],
            "actions": [
                "The payment should reach the account without issue.",
                "The customer expects timely settlement.",
                "The release is part of the regular payment cycle.",
                "The transfer matches the agreed schedule.",
                "No further approvals are presently blocking completion.",
                "The payment path is already approved.",
                "The process is moving forward normally.",
                "The scheduled transfer is already underway.",
            ],
            "suffixes": [
                "We are not seeing a payment disruption at this time.",
                "The invoice remains on track for on-time settlement.",
                "Any delay would be limited to routine processing.",
                "The timing appears consistent with the prior agreement.",
                "The customer has no reason to defer payment.",
                "There is a clear payment plan in place.",
                "This does not look like a default scenario.",
                "The payment commitment appears solid.",
            ],
        },
        {
            "group_id": "LOW_04_NO_BLOCKER",
            "prefixes": [
                "The invoice is approved and there is no payment blocker.",
                "There are currently no issues preventing payment.",
                "The customer has confirmed the invoice and expects no delay.",
                "The account is healthy and the outstanding amount is still on track.",
                "The payment is not facing any material obstacle.",
                "There is nothing in the current state that would prevent settlement.",
                "The finance team reports that the invoice is ready to pay.",
                "The customer is prepared to complete the transfer without change.",
            ],
            "actions": [
                "The payment should proceed normally.",
                "The transaction is expected to clear as scheduled.",
                "The invoice remains in good standing.",
                "The customer has not raised any concerns.",
                "The payment timing is considered routine.",
                "The funds are likely to be released within the agreed window.",
                "The current status suggests no special intervention is needed.",
                "There is no sign of disruption in the payment workflow.",
            ],
            "suffixes": [
                "The customer has not indicated any reason to postpone payment.",
                "This is a routine collection case with no material risk.",
                "The payment plan remains on track.",
                "There is no evidence of a temporary or prolonged delay.",
                "The invoice should be treated as low-risk from a payment standpoint.",
                "The buyer appears willing and able to meet the obligation.",
                "The outstanding amount is still expected to be resolved on time.",
                "This should not impact the expected cash receipt timing.",
            ],
        },
        {
            "group_id": "LOW_05_READY_TO_PAY",
            "prefixes": [
                "The invoice is ready to be paid.",
                "Payment can be released immediately.",
                "The funds are available and the transfer is ready.",
                "The customer has confirmed the invoice is ready for settlement.",
                "The transaction has been approved and is in the release queue.",
                "Treasury confirms the account can send the payment now.",
                "The payment is prepared and awaiting execution.",
                "The customer has moved the invoice into the payment pipeline.",
            ],
            "actions": [
                "This is expected to happen on the agreed date.",
                "The release is a straightforward part of the normal process.",
                "There is no delay indicated by the finance team.",
                "The invoice is clean and ready for release.",
                "The payment should clear without additional review.",
                "No problem is currently visible in the payment chain.",
                "The timing should remain stable.",
                "The payment progression is on schedule.",
            ],
            "suffixes": [
                "The customer is organized and prepared to pay.",
                "This appears to be routine and regular.",
                "There is no sign that settlement will be deferred.",
                "The current status supports a low-risk customer communication.",
                "There is no present risk of nonpayment.",
                "The invoice should be processed without issue.",
                "The account is in a good position for settlement.",
                "No special payment follow-up should be necessary.",
            ],
        },
        {
            "group_id": "LOW_06_CONFIRMED",
            "prefixes": [
                "We have already made the payment.",
                "The transfer was sent successfully.",
                "The invoice has already been paid in full.",
                "The customer has confirmed that the funds were released.",
                "The payment has been processed and the bank confirmation is on file.",
                "This invoice was paid and the remittance has been completed.",
                "The settlement is already reflected in the account.",
                "Payment was executed and the transaction is complete.",
            ],
            "actions": [
                "There is no pending payment action.",
                "The outstanding balance is resolved.",
                "The case does not require any further follow-up.",
                "The invoice is no longer outstanding.",
                "The transfer is complete and verified.",
                "The collection process is effectively finished.",
                "The customer has met its payment obligation.",
                "No additional cash collection effort is needed.",
            ],
            "suffixes": [
                "This should be treated as fully settled.",
                "The customer appears fully current.",
                "There is no unresolved payment issue in this case.",
                "No late-payment risk is present.",
                "The record should show that the invoice is paid.",
                "The account reflects a completed transaction.",
                "The collection team can close the case.",
                "The payment outcome is already confirmed.",
            ],
        },
    ]

    medium_specs = [
        {
            "group_id": "MED_01_APPROVAL_DELAY",
            "prefixes": [
                "Payment is still pending final approval.",
                "The invoice is under review and awaiting approval.",
                "The customer is waiting on internal approval before payment can be released.",
                "The finance team is still reviewing the invoice.",
                "A final approval step is required before the payment can be sent.",
                "The payment is not yet approved but is moving through the internal process.",
                "The invoice has been flagged for additional approval checks.",
                "The customer is completing the internal sign-off needed for settlement.",
            ],
            "actions": [
                "The payment is likely to land within a short delay.",
                "A small postponement is expected while approval completes.",
                "The timing may shift by a few business days.",
                "The transfer should happen once the review step is finished.",
                "There is a reasonable chance of a minor processing delay.",
                "The payment remains possible but not yet locked in.",
                "The customer believes the invoice will eventually be paid.",
                "The release is pending but the situation is manageable.",
            ],
            "suffixes": [
                "The payment should be visible once the administrative hold is cleared.",
                "There is no indication that the invoice has been rejected.",
                "The delay appears to be procedural rather than financial.",
                "The customer expects to meet the obligation after the approval process.",
                "The timeline may slip slightly, but the payment is still intended.",
                "The invoice is not canceled, only waiting on internal completion.",
                "The customer confirms the payment is likely but not yet confirmed.",
                "The account is still expected to receive the funds shortly.",
            ],
        },
        {
            "group_id": "MED_02_RECONCILIATION",
            "prefixes": [
                "The payment is being held for reconciliation.",
                "The customer is finishing a reconciliation check before releasing funds.",
                "The invoice is pending internal reconciliation.",
                "The transfer is delayed until reconciliation is complete.",
                "Finance is verifying account details before processing the payment.",
                "The customer is matching documents before approval is granted.",
                "The invoice is waiting on a reconciliation step.",
                "The payment team is still confirming the invoice details.",
            ],
            "actions": [
                "The invoice is still likely to be paid, but the timing is uncertain.",
                "A brief delay is expected while the file is completed.",
                "There may be a moderate postponement before release.",
                "The process is moving forward but not yet final.",
                "The payment should occur after the outstanding administrative tasks are closed.",
                "The customer expects to settle once the review is complete.",
                "The amount remains committed but needs the final approval stage.",
                "The transfer is still on the roadmap but needs the paperwork to finish.",
            ],
            "suffixes": [
                "We have not seen evidence of refusal, only a processing pause.",
                "The payment date is not fully fixed yet.",
                "The invoice is still considered recoverable.",
                "The delay is tied to internal confirmation rather than a hard repudiation.",
                "The customer still expects to settle once the review task is completed.",
                "This appears to be a temporary administrative delay.",
                "The payment is being worked through the normal operations flow.",
                "The likely outcome is eventual payment, but the exact date is still pending.",
            ],
        },
        {
            "group_id": "MED_03_SHORT_DELAY",
            "prefixes": [
                "We are expecting a short delay while the payment is processed.",
                "There is a temporary slowdown in the payment cycle.",
                "The customer says the transfer will be made later than originally planned.",
                "The invoice is delayed because the payment batch is still running.",
                "The payment has been queued but may take a little longer than usual.",
                "The customer is facing a brief operational delay in payment execution.",
                "The transfer is delayed by normal processing timing.",
                "The payment will likely come through after the current batch closes.",
            ],
            "actions": [
                "The customer expects payment to land within a few days.",
                "The payment is still intended and should come through soon.",
                "The delay is temporary and tied to current operations.",
                "The amount remains due and the payment plan is still active.",
                "The customer is hopeful the invoice will be settled shortly.",
                "The customer is not refusing the invoice, but timing may slip slightly.",
                "The payment is not cancelled, only deferred for a short period.",
                "The case remains open and processing is ongoing.",
            ],
            "suffixes": [
                "The invoice still has a realistic path to payment.",
                "The delay should not be interpreted as a complete nonpayment.",
                "The customer appears willing to pay with a modest push-out.",
                "The outstanding amount is still expected to be settled in the near term.",
                "The account may show a short delay, but not a permanent issue.",
                "The customer has not rejected the invoice, only shifted the timing.",
                "The situation is manageable but needs close monitoring.",
                "There is not yet enough evidence to classify the account as high risk.",
            ],
        },
        {
            "group_id": "MED_04_INTERNAL_REVIEW",
            "prefixes": [
                "The invoice is undergoing internal review.",
                "The customer is reviewing the invoice before release.",
                "The payment is being held pending internal checks.",
                "The customer needs a little more time to complete the internal review.",
                "A review step is currently in progress before disbursement.",
                "The payment team has not finalized the transfer yet.",
                "The invoice is going through the standard internal review process.",
                "The customer is still verifying the invoice details before payment.",
            ],
            "actions": [
                "The expected outcome is still a payment, but the schedule is uncertain.",
                "The amount remains owed and the buyer is engaged.",
                "The terms are still being worked through.",
                "The transfer is likely but the date remains flexible.",
                "There is a moderate chance of delay while the review concludes.",
                "The customer has not indicated disruption, only continued review.",
                "The payment may move later than originally intended.",
                "The case still looks recoverable, but the timing is not final.",
            ],
            "suffixes": [
                "The account should be watched closely over the next few days.",
                "This is a genuine delay risk, but not necessarily a total default.",
                "The customer has not refused payment, only slowed the process.",
                "The invoice remains in play, but the date is still uncertain.",
                "The payment status should be reevaluated after the review finishes.",
                "The risk profile is elevated compared with a fully settled invoice.",
                "A few more administrative steps are necessary before the payment can be released.",
                "The case should be handled as a moderate risk until payment is confirmed.",
            ],
        },
        {
            "group_id": "MED_05_PENDING_BUT_REASONABLE",
            "prefixes": [
                "The payment is outstanding but still expected.",
                "The customer says they are still planning to pay.",
                "The invoice remains open, and the buyer has not yet finalized the transfer.",
                "The payment was not completed yet, but the customer is still engaged.",
                "The amount remains due and the customer has not closed the file.",
                "The invoice is still pending a release decision.",
                "The customer has not confirmed the payment date, only their intent to pay.",
                "The payment status is unresolved but the customer is still communicating.",
            ],
            "actions": [
                "The most likely outcome is payment after a short additional delay.",
                "The customer still appears willing to settle, but timing remains unclear.",
                "The situation suggests a moderate payment risk rather than immediate default.",
                "The account needs continued monitoring because the release date is not yet fixed.",
                "The buyer is aware of the obligation and is still working through it.",
                "A delayed release is likely, but the underlying contract remains active.",
                "Settlement still appears possible if the internal process moves smoothly.",
                "The invoice remains payable, just not yet confirmed in cash.",
            ],
            "suffixes": [
                "The customer is not saying no, only that timing is still being worked out.",
                "There is still a credible path to payment.",
                "The customer has not outright refused the invoice.",
                "The case looks moderate rather than severe.",
                "A short extension is the most plausible outcome.",
                "The amount should still be treated as recoverable but not yet settled.",
                "The buyer is engaged, but the payment date remains provisional.",
                "The current information supports caution rather than escalation.",
            ],
        },
        {
            "group_id": "MED_06_TEMPORARY_ISSUE",
            "prefixes": [
                "The customer is experiencing a temporary issue that may affect payment timing.",
                "There is a short-term obstacle affecting the payment process.",
                "The invoice is delayed by a temporary operational issue.",
                "The customer is dealing with a brief constraint that could shift the transfer.",
                "The payment is still expected, but a temporary problem is slowing the process.",
                "The buyer says the payment will likely occur after a short disruption.",
                "The customer has a temporary issue preventing immediate settlement.",
                "There is a current operational bottleneck affecting payment release.",
            ],
            "actions": [
                "The customer still intends to pay, but the date is not guaranteed yet.",
                "The payment timeline is under review and likely to move.",
                "The issue seems temporary, but the payment may arrive later than anticipated.",
                "The case is unresolved and should be followed closely.",
                "The amount is still expected, but no fixed date has been confirmed.",
                "The customer has not canceled the invoice, only rescheduled the release.",
                "The payment path is still alive, though delayed.",
                "A short-term interruption is currently the main concern.",
            ],
            "suffixes": [
                "The customer remains engaged and has not closed the account.",
                "The payment might still happen, but not necessarily on the original date.",
                "This is an operational delay rather than a full repudiation.",
                "The risk is meaningful but not yet acute.",
                "The customer is still in communication and wants to resolve the invoice.",
                "The situation requires a moderate-risk handling strategy.",
                "The payment event is delayed but not abandoned.",
                "The case should remain in watchlist status until release is confirmed.",
            ],
        },
    ]

    high_specs = [
        {
            "group_id": "HIGH_01_NO_DATE",
            "prefixes": [
                "We cannot confirm when the payment will be made.",
                "The customer has not provided a firm settlement date.",
                "There is currently no confirmed payment date.",
                "The payment timing remains uncertain.",
                "The customer cannot commit to a payment date right now.",
                "The invoice is still open and the release date has not been set.",
                "The customer is unable to provide a reliable payment schedule.",
                "The payment date is still unresolved.",
            ],
            "actions": [
                "This raises a serious concern about when the invoice will be settled.",
                "The buyer has not provided the necessary assurance that the amount will be paid.",
                "The account should be treated as highly uncertain until payment is verified.",
                "The customer may be unable to release the funds in the near term.",
                "There is a substantial chance that payment will move far beyond the original due date.",
                "The invoice remains at risk because the customer has not committed to a date.",
                "The lack of a confirmed schedule suggests payment may be delayed materially.",
                "The situation should be treated as serious until the cash movement is confirmed.",
            ],
            "suffixes": [
                "The customer appears unable or unwilling to commit.",
                "This is a meaningful nonpayment risk scenario.",
                "The payment outcome is not yet dependable.",
                "The invoice should be escalated for urgent follow-up.",
                "The risk of a prolonged delay is high.",
                "There is no visible payment certainty at this point.",
                "The amount may not be collected on time.",
                "The customer has not established a credible near-term settlement plan.",
            ],
        },
        {
            "group_id": "HIGH_02_CASHFLOW",
            "prefixes": [
                "We are facing a temporary liquidity problem.",
                "The customer is dealing with a cash-flow issue that is affecting payment.",
                "The business is currently constrained by limited liquidity.",
                "The customer is experiencing a short-term cash shortage.",
                "There is a temporary funding problem preventing immediate settlement.",
                "The customer is unable to make the transfer today because of cash constraints.",
                "The company is currently constrained by cash-flow pressure.",
                "The payment is being delayed because of liquidity concerns.",
            ],
            "actions": [
                "The invoice is at risk because the customer cannot confirm a repayment date.",
                "The payment may be pushed back materially while the shortfall is managed.",
                "The customer is unable to promise a normal payment schedule.",
                "The account likely needs a more urgent collection strategy.",
                "There is a strong likelihood that settlement will be late.",
                "The customer has admitted an ongoing financial constraint.",
                "The collection team should expect a difficult and delayed outcome.",
                "The customer has not provided an immediate path to settlement.",
            ],
            "suffixes": [
                "This should be treated as a serious payment risk.",
                "The cash-flow disruption may result in a prolonged delay.",
                "The invoice is being threatened by a real financial limitation.",
                "No reliable date can be given at this stage.",
                "The risk of nonpayment is elevated until the situation changes.",
                "This case is not simply administrative; there is a genuine cash issue.",
                "The customer is signaling real strain in meeting the obligation.",
                "The payment will likely be delayed until liquidity improves.",
            ],
        },
        {
            "group_id": "HIGH_03_BLOCKED",
            "prefixes": [
                "The payment has been blocked.",
                "The invoice cannot be paid at this time.",
                "The transfer has been put on hold.",
                "Payment has been suspended pending further review.",
                "The payment release is currently blocked by internal constraints.",
                "The customer cannot proceed with the payment right now.",
                "The invoice is effectively stalled because the transfer has been blocked.",
                "The payment is not proceeding because the customer is unable to release funds.",
            ],
            "actions": [
                "The buyer is not able to provide a realistic payment date.",
                "The invoice is at serious risk of a major delay.",
                "The customer has essentially paused the obligation.",
                "The payment may not be possible until the blocking issue is resolved.",
                "The case requires immediate follow-up and escalation.",
                "The payment path appears compromised.",
                "The amount is not currently in motion.",
                "There is a strong likelihood that the invoice will remain unpaid for a long period.",
            ],
            "suffixes": [
                "This is a high-risk nonpayment scenario.",
                "There is no immediate evidence of a viable recovery path.",
                "The customer has not established a credible payment plan.",
                "The risk of prolonged default is significant.",
                "The invoice should be managed as a blocked account.",
                "This situation warrants urgent commercial action.",
                "The customer is not close to making a payment decision.",
                "A sustained delay appears likely.",
            ],
        },
        {
            "group_id": "HIGH_04_DISPUTE",
            "prefixes": [
                "The payment is being withheld because of a dispute.",
                "The customer is refusing payment while the invoice is being contested.",
                "The customer disputes the invoice and will not release funds until the issue is resolved.",
                "The payment is paused due to a disagreement over the invoice details.",
                "The customer says the invoice is under dispute and therefore no payment will be made.",
                "The transfer is blocked because the customer is disputing the charge.",
                "The invoice is currently contested and payment cannot proceed.",
                "The customer is unwilling to pay until the dispute is resolved.",
            ],
            "actions": [
                "This indicates a significant risk of missed or prolonged payment.",
                "The buyer may be unable to settle until the dispute is closed.",
                "The invoice has become a high-risk collection matter.",
                "The customer is not confirming any immediate payment timeline.",
                "The dispute introduces a real barrier to collection.",
                "The customer is effectively withholding funds until the issue is clarified.",
                "The payment date remains uncertain and likely to move back significantly.",
                "This is not a routine case and should be treated seriously.",
            ],
            "suffixes": [
                "The customer is not acknowledging the invoice as payable.",
                "This should be classified as high risk.",
                "The financial outcome is uncertain and potentially adverse.",
                "The case may require dispute resolution before any cash movement occurs.",
                "The relationship is currently strained and the amount may not be recovered quickly.",
                "No payment can be expected until the dispute is addressed.",
                "The account appears to have material unresolved issues.",
                "The outstanding invoice may remain unpaid for an extended period.",
            ],
        },
        {
            "group_id": "HIGH_05_PROLONGED_DELAY",
            "prefixes": [
                "The customer expects the payment to be delayed significantly beyond the original due date.",
                "The invoice will likely be paid only after an extended period.",
                "The customer is unable to meet the normal settlement window and expects a major delay.",
                "The payment is likely to slip well beyond the due date.",
                "The customer has indicated that settlement will be much later than agreed.",
                "The transfer will not be made within the current schedule.",
                "The invoice is facing a serious schedule extension.",
                "The customer is forecasting a prolonged delay rather than a normal payment cycle.",
            ],
            "actions": [
                "The invoice should be treated as high risk until payment is actually received.",
                "The customer has signaled a materially worse payment outcome.",
                "The account should be elevated because the delay is substantial.",
                "The customer has not offered a firm near-term date.",
                "The chance of a timely collection is low in this scenario.",
                "This is not a short process delay but an extended postponement.",
                "The expected cash timing is far outside the normal expectation.",
                "The outstanding amount is likely to remain open for a prolonged period.",
            ],
            "suffixes": [
                "The business is effectively saying the payment may be late for a long time.",
                "The invoice should be monitored as high risk.",
                "No immediate recovery should be assumed.",
                "The amount could remain unsettled for a meaningful stretch.",
                "This is materially worse than a routine administrative delay.",
                "The payment could be pushed far beyond the original timeline.",
                "The customer is not demonstrating a credible near-term path to payment.",
                "The collection team should plan for significant delay and limited immediate recovery.",
            ],
        },
        {
            "group_id": "HIGH_06_FAILURE_TO_COMMIT",
            "prefixes": [
                "The customer cannot provide a reliable payment commitment.",
                "The buyer is not prepared to confirm when the invoice will be paid.",
                "The customer has not offered a credible commitment to settlement.",
                "The business is unable to state a dependable payment timeline.",
                "The customer cannot confirm whether the invoice will be paid on time.",
                "The organization is not in a position to guarantee payment.",
                "The account currently lacks a trustworthy repayment plan.",
                "The customer has not given a dependable statement about when cash will move.",
            ],
            "actions": [
                "This suggests a serious risk of missed or delayed payment.",
                "The obligation is not being honored with enough clarity or certainty.",
                "The customer has not demonstrated the ability to meet the invoice promptly.",
                "The payment horizon remains highly uncertain.",
                "A prolonged delay or nonpayment appears plausible.",
                "We should assume the account requires immediate attention.",
                "The customer is not providing the level of confidence needed for low-risk treatment.",
                "The invoice should be considered at high risk until payment is actually received.",
            ],
            "suffixes": [
                "The payment outcome is not dependable today.",
                "The customer is not giving enough assurance to classify this as low or medium risk.",
                "The invoice may remain unsettled for an extended period.",
                "The current communication is insufficient to support a strong recovery expectation.",
                "The business is effectively signaling uncertainty about settlement.",
                "This should not be treated as a routine payment case.",
                "The likelihood of a prolonged delay remains high.",
                "The amount should be managed with a high-risk posture.",
            ],
        },
    ]

    all_specs = []
    for label, specs in [("LOW_RISK", low_specs), ("MEDIUM_RISK", medium_specs), ("HIGH_RISK", high_specs)]:
        for spec in specs:
            all_specs.append({"label": label, **spec})

    family_rows = []
    rng = random.Random(SEED)
    for family in all_specs:
        text_rows = []
        for text in sample_product(family["prefixes"], family["actions"], family["suffixes"], limit=165):
            text_rows.append(
                {
                    "text": text,
                    "label": family["label"],
                    "template_group_id": family["group_id"],
                }
            )
        family_rows.extend(text_rows)

    df = pd.DataFrame(family_rows)
    df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)
    return df


def split_groups(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = random.Random(SEED)
    groups_by_label = {label: sorted(df[df["label"] == label]["template_group_id"].unique()) for label in LABEL_ORDER}

    train_groups, val_groups, test_groups = [], [], []
    for label in LABEL_ORDER:
        groups = groups_by_label[label]
        rng.shuffle(groups)
        total = len(groups)
        train_count = max(1, int(total * 0.70))
        val_count = max(1, int(total * 0.15))
        train_groups.extend(groups[:train_count])
        val_groups.extend(groups[train_count : train_count + val_count])
        test_groups.extend(groups[train_count + val_count :])

    train_df = df[df["template_group_id"].isin(train_groups)].copy()
    val_df = df[df["template_group_id"].isin(val_groups)].copy()
    test_df = df[df["template_group_id"].isin(test_groups)].copy()

    return train_df, val_df, test_df


def make_vectorizer(use_char_ngrams: bool) -> TfidfVectorizer | FeatureUnion:
    if not use_char_ngrams:
        return TfidfVectorizer(
            ngram_range=(1, 2),
            sublinear_tf=True,
            min_df=1,
            strip_accents="unicode",
            norm="l2",
            max_features=6000,
        )

    return FeatureUnion(
        transformer_list=[
            (
                "word",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    sublinear_tf=True,
                    min_df=1,
                    strip_accents="unicode",
                    norm="l2",
                    max_features=5000,
                ),
            ),
            (
                "char",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(3, 5),
                    sublinear_tf=True,
                    min_df=1,
                    strip_accents="unicode",
                    norm="l2",
                    max_features=4000,
                ),
            ),
        ]
    )


def build_model_specs():
    specs = {}
    specs["A_tfidf_logreg"] = Pipeline(
        [
            ("vectorizer", make_vectorizer(use_char_ngrams=False)),
            (
                "classifier",
                LogisticRegression(
                    max_iter=5000,
                    C=3.0,
                    class_weight="balanced",
                    random_state=SEED,
                    multi_class="auto",
                ),
            ),
        ]
    )
    specs["B_word_char_logreg"] = Pipeline(
        [
            ("vectorizer", make_vectorizer(use_char_ngrams=True)),
            (
                "classifier",
                LogisticRegression(
                    max_iter=5000,
                    C=2.5,
                    class_weight="balanced",
                    random_state=SEED,
                    multi_class="auto",
                ),
            ),
        ]
    )
    specs["C_tfidf_calibrated_svc"] = Pipeline(
        [
            ("vectorizer", make_vectorizer(use_char_ngrams=False)),
            (
                "classifier",
                CalibratedClassifierCV(
                    estimator=LinearSVC(C=1.5, class_weight="balanced", random_state=SEED),
                    cv=3,
                    method="sigmoid",
                ),
            ),
        ]
    )
    specs["D_word_char_calibrated_svc"] = Pipeline(
        [
            ("vectorizer", make_vectorizer(use_char_ngrams=True)),
            (
                "classifier",
                CalibratedClassifierCV(
                    estimator=LinearSVC(C=1.0, class_weight="balanced", random_state=SEED),
                    cv=3,
                    method="sigmoid",
                ),
            ),
        ]
    )
    return specs


def compute_multiclass_metrics(y_true: Iterable[str], y_pred: Iterable[str], y_prob: np.ndarray) -> dict:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_prob = np.asarray(y_prob)

    if y_prob.ndim == 1:
        y_prob = np.vstack([1 - y_prob, y_prob]).T

    labels = LABEL_ORDER
    result = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "roc_auc_ovr": float(roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")),
        "pr_auc_macro": float(average_precision_score(y_true, y_prob, average="macro")),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "per_class": {},
        "support": {},
    }

    for label in labels:
        label_index = labels.index(label)
        mask = y_true == label
        class_y_true = (y_true == label).astype(int)
        class_y_pred = (y_pred == label).astype(int)
        result["per_class"][label] = {
            "precision": float(precision_score(class_y_true, class_y_pred, zero_division=0)),
            "recall": float(recall_score(class_y_true, class_y_pred, zero_division=0)),
            "f1": float(f1_score(class_y_true, class_y_pred, zero_division=0)),
            "support": int(mask.sum()),
        }
        result["support"][label] = int(mask.sum())

    return result


def score_thresholds(y_true: pd.Series, y_prob: np.ndarray, labels: List[str]) -> tuple[dict, dict]:
    high_candidates = np.linspace(0.30, 0.90, 13)
    med_candidates = np.linspace(0.20, 0.80, 13)

    best = None
    best_thresholds = {"high": 0.5, "medium": 0.35}

    for high_t in high_candidates:
        for med_t in med_candidates:
            pred = []
            for row in y_prob:
                probs = dict(zip(labels, row))
                high_p = probs.get("HIGH_RISK", 0.0)
                med_p = probs.get("MEDIUM_RISK", 0.0)
                if high_p >= high_t:
                    pred.append("HIGH_RISK")
                elif med_p >= med_t:
                    pred.append("MEDIUM_RISK")
                else:
                    pred.append("LOW_RISK")
            f1 = f1_score(y_true, pred, labels=labels, average="macro", zero_division=0)
            if best is None or f1 > best:
                best = f1
                best_thresholds = {"high": float(high_t), "medium": float(med_t)}

    return best_thresholds, {
        "validation_f1_macro": float(best) if best is not None else 0.0,
    }


def apply_thresholds(probabilities: np.ndarray, thresholds: dict, labels: List[str]) -> np.ndarray:
    preds = []
    for row in probabilities:
        probs = dict(zip(labels, row))
        high_p = probs.get("HIGH_RISK", 0.0)
        med_p = probs.get("MEDIUM_RISK", 0.0)
        if high_p >= thresholds["high"]:
            preds.append("HIGH_RISK")
        elif med_p >= thresholds["medium"]:
            preds.append("MEDIUM_RISK")
        else:
            preds.append("LOW_RISK")
    return np.asarray(preds)


def build_challenge_set(data_dir: Path) -> pd.DataFrame:
    challenge_specs = [
        {
            "group_id": "CHALLENGE_01_NEGATED",
            "label": "LOW_RISK",
            "texts": [
                "We do not expect any delay; the payment has already been transferred.",
                "The transfer is not delayed, and the invoice is already settled.",
                "The payment was not missed; the funds were released earlier than expected.",
                "There is no outstanding issue because the transfer was already completed.",
                "The customer is not holding back payment; the invoice is already paid.",
            ],
        },
        {
            "group_id": "CHALLENGE_02_INDIRECT",
            "label": "LOW_RISK",
            "texts": [
                "Our accounts team advises that the amount will be sent on the agreed date, so there is no need for escalation.",
                "The invoice is in good standing, and the expected transfer is already aligned with the payment window.",
                "The business reports a normal processing cycle, with no indication that settlement will slip.",
                "There is no material obstacle; the payment route is already approved and the transfer is proceeding.",
                "The customer has confirmed the invoice is compliant and the release is expected within the standard schedule.",
            ],
        },
        {
            "group_id": "CHALLENGE_03_UNCERTAIN",
            "label": "MEDIUM_RISK",
            "texts": [
                "The invoice is still with finance, but it appears likely the payment will be made after a brief review.",
                "The team is still tightening the final approval step; settlement may move by a few days.",
                "The transfer has not yet been executed, though the customer says the amount is still intended.",
                "We are waiting on a final check, and the payment date remains tentative rather than fixed.",
                "The invoice is moving through the normal process, but the exact date is still being confirmed.",
            ],
        },
        {
            "group_id": "CHALLENGE_04_LIQUIDITY",
            "label": "HIGH_RISK",
            "texts": [
                "The business is dealing with a cash squeeze and cannot promise a near-term release.",
                "Liquidity remains constrained, so the payment date is currently unavailable.",
                "The customer is under short-term financial pressure and cannot provide a reliable settlement path.",
                "The company has limited cash and is not in a position to commit to when the invoice will be paid.",
                "The payment may be delayed until the liquidity situation improves materially.",
            ],
        },
        {
            "group_id": "CHALLENGE_05_SHORT_MSG",
            "label": "HIGH_RISK",
            "texts": [
                "No confirmed date.",
                "Cash flow issue.",
                "Payment blocked.",
                "Unable to settle right now.",
                "No firm payout window.",
            ],
        },
    ]

    rows = []
    for spec in challenge_specs:
        for text in spec["texts"]:
            rows.append({"text": text, "label": spec["label"], "template_group_id": spec["group_id"]})

    challenge_df = pd.DataFrame(rows)
    challenge_df = challenge_df.sample(frac=1, random_state=SEED).reset_index(drop=True)

    challenge_path = data_dir / "nlp_unseen_test.csv"
    challenge_df.to_csv(challenge_path, index=False)
    return challenge_df


def load_or_build_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data_path = DATA_DIR / "nlp_synthetic_dataset.csv"
    if data_path.exists():
        df = pd.read_csv(data_path)
    else:
        df = build_nlp_candidate_sets()
        df.to_csv(data_path, index=False)

    train_df, val_df, test_df = split_groups(df)
    challenge_df = build_challenge_set(DATA_DIR)
    return train_df, val_df, test_df, challenge_df


def fit_and_evaluate_model(
    model_name: str,
    model: Pipeline,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    challenge_df: pd.DataFrame,
) -> dict:
    X_train = train_df["text"].tolist()
    y_train = train_df["label"].tolist()
    X_val = val_df["text"].tolist()
    y_val = val_df["label"].tolist()
    X_test = test_df["text"].tolist()
    y_test = test_df["label"].tolist()
    X_challenge = challenge_df["text"].tolist()
    y_challenge = challenge_df["label"].tolist()

    model.fit(X_train, y_train)

    classifier = model.named_steps["classifier"]
    classes = list(classifier.classes_)
    probability_order = classes

    val_probability = model.predict_proba(X_val)
    val_predictions = model.predict(X_val)

    thresholds, threshold_summary = score_thresholds(val_df["label"], val_probability, probability_order)
    val_threshold_predictions = apply_thresholds(val_probability, thresholds, probability_order)

    val_metrics = compute_multiclass_metrics(y_val, val_threshold_predictions, val_probability)
    val_metrics["thresholds"] = thresholds

    test_probability = model.predict_proba(X_test)
    test_predictions = apply_thresholds(test_probability, thresholds, probability_order)
    test_metrics = compute_multiclass_metrics(y_test, test_predictions, test_probability)
    test_metrics["thresholds"] = thresholds

    challenge_probability = model.predict_proba(X_challenge)
    challenge_predictions = apply_thresholds(challenge_probability, thresholds, probability_order)
    challenge_metrics = compute_multiclass_metrics(y_challenge, challenge_predictions, challenge_probability)
    challenge_metrics["thresholds"] = thresholds

    model_path = MODELS_DIR / f"nlp_{model_name}.joblib"
    joblib.dump(model, model_path)

    return {
        "model_name": model_name,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "challenge_metrics": challenge_metrics,
        "thresholds": thresholds,
        "summary": {
            "val_f1_macro": val_metrics["f1_macro"],
            "test_f1_macro": test_metrics["f1_macro"],
            "challenge_f1_macro": challenge_metrics["f1_macro"],
            "val_high_risk_recall": val_metrics["per_class"]["HIGH_RISK"]["recall"],
            "test_high_risk_recall": test_metrics["per_class"]["HIGH_RISK"]["recall"],
            "challenge_high_risk_recall": challenge_metrics["per_class"]["HIGH_RISK"]["recall"],
        },
    }


def write_reports(best_result: dict, results: List[dict]) -> None:
    comparison_rows = []
    for result in results:
        for split_name in ["val_metrics", "test_metrics", "challenge_metrics"]:
            metrics = result[split_name]
            comparison_rows.append(
                {
                    "model": result["model_name"],
                    "split": split_name.replace("_metrics", ""),
                    "accuracy": round(metrics["accuracy"], 6),
                    "precision_macro": round(metrics["precision_macro"], 6),
                    "recall_macro": round(metrics["recall_macro"], 6),
                    "f1_macro": round(metrics["f1_macro"], 6),
                    "roc_auc_ovr": round(metrics["roc_auc_ovr"], 6),
                    "pr_auc_macro": round(metrics["pr_auc_macro"], 6),
                    "high_risk_recall": round(metrics["per_class"]["HIGH_RISK"]["recall"], 6),
                    "high_risk_precision": round(metrics["per_class"]["HIGH_RISK"]["precision"], 6),
                    "low_risk_recall": round(metrics["per_class"]["LOW_RISK"]["recall"], 6),
                    "medium_risk_recall": round(metrics["per_class"]["MEDIUM_RISK"]["recall"], 6),
                    "confusion_matrix": metrics["confusion_matrix"],
                    "thresholds": json.dumps(metrics["thresholds"]),
                }
            )

    comparison_df = pd.DataFrame(comparison_rows)
    comparison_df.to_csv(REPORTS_DIR / "nlp_model_comparison.csv", index=False)

    best_model_name = best_result["model_name"]
    best_model = joblib.load(MODELS_DIR / f"nlp_{best_model_name}.joblib")
    joblib.dump(best_model, MODELS_DIR / "nlp_payment_risk.joblib")

    # Save vectorizer separately for inspection/backup, but keep the full pipeline in the main artifact.
    if "vectorizer" in best_model.named_steps:
        joblib.dump(best_model.named_steps["vectorizer"], MODELS_DIR / "nlp_vectorizer.joblib")

    best_summary = {
        "best_model": best_model_name,
        "selected_thresholds": best_result["thresholds"],
        "validation": best_result["val_metrics"],
        "test": best_result["test_metrics"],
        "challenge": best_result["challenge_metrics"],
        "overfitting_signal": {
            "val_minus_test_f1_macro": round(best_result["val_metrics"]["f1_macro"] - best_result["test_metrics"]["f1_macro"], 6),
            "test_minus_challenge_f1_macro": round(best_result["test_metrics"]["f1_macro"] - best_result["challenge_metrics"]["f1_macro"], 6),
        },
    }

    (REPORTS_DIR / "nlp_metrics.json").write_text(json.dumps(best_summary, indent=2), encoding="utf-8")
    (REPORTS_DIR / "nlp_thresholds.json").write_text(json.dumps(best_result["thresholds"], indent=2), encoding="utf-8")


def write_audit_report(results: List[dict], best_result: dict) -> None:
    best_name = best_result["model_name"]
    report_lines = [
        "# NLP Audit Report",
        "",
        "## Executive summary",
        "",
        "The original NLP component in this repository was a small synthetic/demo-only TF-IDF + Logistic Regression classifier built from a handful of templated examples. That design made it vulnerable to lexical dependence, duplicated message patterns, and poor generalization to unseen paraphrases.",
        "",
        "## Findings from the current implementation",
        "",
        "1. The NLP model was trained from `scripts/build_all_models.py` from a small synthetic dataset with only a few dozen repeated patterns, not from real customer payment messages.",
        "2. The previous synthetic corpus reused the same sentence structure across many examples. That creates train/test contamination risk and weak real-world generalization.",
        "3. The original model used a binary high-risk-vs-low-risk decision with a single probability threshold, which is not well matched to a three-level risk interpretation (`LOW`, `MEDIUM`, `HIGH`).",
        "4. The old design did not perform group-aware splitting, so semantically similar templates could leak across train/validation/test.",
        "5. The original model did not include calibration or threshold tuning on a separate validation set, so probability estimates were not robustly checked.",
        "6. The current model was optional/demo-only and clearly documented as such; it should never be described as a production model trained on invoice data.",
        "",
        "## What changed",
        "",
        "- Replaced the small synthetic corpus with a larger, grouped synthetic dataset designed to preserve semantic intent while reducing template leakage.",
        "- Added a held-out challenge set of semantically difficult, unseen payment messages for honest evaluation.",
        "- Compared multiple TF-IDF-based baselines, including word-only, word+char, and calibrated linear SVM variants.",
        "- Added explicit threshold optimization on validation data, then froze those thresholds for test and challenge evaluation.",
        "- Saved the best validated pipeline as `models/nlp_payment_risk.joblib` and kept a vectorizer artifact as `models/nlp_vectorizer.joblib` for transparency.",
        "",
        "## Model comparison summary",
        "",
        "| Model | Validation Macro F1 | Test Macro F1 | Challenge Macro F1 | High-risk recall |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]

    for result in results:
        report_lines.append(
            f"| {result['model_name']} | {result['val_metrics']['f1_macro']:.4f} | {result['test_metrics']['f1_macro']:.4f} | {result['challenge_metrics']['f1_macro']:.4f} | {result['challenge_metrics']['per_class']['HIGH_RISK']['recall']:.4f} |"
        )

    report_lines.extend(
        [
            "",
            "## Best validated candidate",
            "",
            f"The best-performing model selected from validation performance was `{best_name}`.",
            "",
            "## Recommendations",
            "",
            "- Keep the existing app architecture, but treat the NLP model as a separate payment-message signal rather than as a full invoice risk engine.",
            "- Swap in real annotated payment messages when available; the current dataset is still synthetic and should be treated as a demo-grade improvement over the original baseline.",
            "- Continue evaluating on truly unseen customer messages and use challenge-set performance as a guardrail against overfitting.",
            "- If a production-labelled corpus becomes available, retrain and revalidate with the same group-aware split strategy.",
        ]
    )

    (REPORTS_DIR / "nlp_audit_report.md").write_text("\n".join(report_lines), encoding="utf-8")


def main() -> None:
    train_df, val_df, test_df, challenge_df = load_or_build_data()

    models = build_model_specs()
    results = []
    for model_name, model in models.items():
        results.append(
            fit_and_evaluate_model(
                model_name=model_name,
                model=model,
                train_df=train_df,
                val_df=val_df,
                test_df=test_df,
                challenge_df=challenge_df,
            )
        )

    best_result = max(
        results,
        key=lambda item: (
            item["val_metrics"]["f1_macro"],
            item["challenge_metrics"]["f1_macro"],
            item["summary"]["val_high_risk_recall"],
        ),
    )

    write_reports(best_result, results)
    write_audit_report(results, best_result)

    print("NLP evaluation complete.")
    print(json.dumps(best_result["summary"], indent=2))


if __name__ == "__main__":
    main()
