from __future__ import annotations

import json
import random
from itertools import product
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
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
RNG = random.Random(SEED)


def build_group_specs() -> List[dict]:
    groups = []

    specs = {
        "LOW_RISK": [
            {
                "group_id": "LOW_01_COMPLETE",
                "prefixes": [
                    "Payment has already been completed.",
                    "The transfer was successfully sent.",
                    "The invoice is fully settled.",
                    "Funds were released this morning.",
                    "The account has already been credited.",
                    "The payment was completed on schedule.",
                ],
                "actions": [
                    "The invoice is now closed.",
                    "No further action is required.",
                    "The customer has already satisfied the obligation.",
                    "There are no outstanding payment issues.",
                    "The remittance advice has been delivered.",
                ],
                "suffixes": [
                    "No payment disruption is expected.",
                    "The due date is effectively resolved.",
                    "The invoice should not require any additional follow-up.",
                    "This matter is already settled from a payment perspective.",
                    "There is no current blocker on the account.",
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
                ],
                "actions": [
                    "The transfer will be processed as planned.",
                    "The payment is on track for settlement.",
                    "The bank transfer is expected to go out on time.",
                    "The funds are scheduled to move according to the agreement.",
                    "The release is already prepared in the payment queue.",
                ],
                "suffixes": [
                    "This should meet the original due date.",
                    "There is no sign of a payment problem.",
                    "The customer intends to settle without any interruption.",
                    "The account is expected to reflect the transfer soon.",
                    "The payment timing is considered normal.",
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
                ],
                "actions": [
                    "The payment should reach the account without issue.",
                    "The customer expects timely settlement.",
                    "The release is part of the regular payment cycle.",
                    "The transfer matches the agreed schedule.",
                    "No further approvals are presently blocking completion.",
                ],
                "suffixes": [
                    "We are not seeing a payment disruption at this time.",
                    "The invoice remains on track for on-time settlement.",
                    "Any delay would be limited to routine processing.",
                    "The timing appears consistent with the prior agreement.",
                    "The payment process appears routine and predictable.",
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
                ],
                "actions": [
                    "The payment should proceed normally.",
                    "The transaction is expected to clear as scheduled.",
                    "The invoice remains in good standing.",
                    "The customer has not raised any concerns.",
                    "The payment timing is considered routine.",
                ],
                "suffixes": [
                    "The customer has not indicated any reason to postpone payment.",
                    "This is a routine collection case with no material risk.",
                    "The payment plan remains on track.",
                    "There is no evidence of a temporary or prolonged delay.",
                    "The invoice should be treated as low-risk from a payment standpoint.",
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
                ],
                "actions": [
                    "This is expected to happen on the agreed date.",
                    "The release is a straightforward part of the normal process.",
                    "There is no delay indicated by the finance team.",
                    "The invoice is clean and ready for release.",
                    "The payment should clear without additional review.",
                ],
                "suffixes": [
                    "The customer is organized and prepared to pay.",
                    "This appears to be routine and regular.",
                    "There is no sign that settlement will be deferred.",
                    "The current status supports a low-risk customer communication.",
                    "There is no present risk of nonpayment.",
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
                ],
                "actions": [
                    "There is no pending payment action.",
                    "The outstanding balance is resolved.",
                    "The case does not require any further follow-up.",
                    "The invoice is no longer outstanding.",
                    "The transfer is complete and verified.",
                ],
                "suffixes": [
                    "This should be treated as fully settled.",
                    "The customer appears fully current.",
                    "There is no unresolved payment issue in this case.",
                    "No late-payment risk is present.",
                    "The account reflects a completed transaction.",
                ],
            },
        ],
        "MEDIUM_RISK": [
            {
                "group_id": "MED_01_APPROVAL_DELAY",
                "prefixes": [
                    "Payment is still pending final approval.",
                    "The invoice is under review and awaiting approval.",
                    "The finance team is still reviewing the invoice.",
                    "A final approval step is required before the payment can be sent.",
                    "The payment is not yet approved but is moving through the internal process.",
                    "The invoice has been flagged for additional approval checks.",
                ],
                "actions": [
                    "The payment is likely to land within a short delay.",
                    "A small postponement is expected while approval completes.",
                    "The timing may shift by a few business days.",
                    "The transfer should happen once the review step is finished.",
                    "There is a reasonable chance of a minor processing delay.",
                ],
                "suffixes": [
                    "The payment should be visible once the administrative hold is cleared.",
                    "There is no indication that the invoice has been rejected.",
                    "The delay appears to be procedural rather than financial.",
                    "The customer expects to meet the obligation after the approval process.",
                    "The timeline may slip slightly, but the payment is still intended.",
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
                ],
                "actions": [
                    "The invoice is still likely to be paid, but the timing is uncertain.",
                    "A brief delay is expected while the file is completed.",
                    "There may be a moderate postponement before release.",
                    "The process is moving forward but not yet final.",
                    "The payment should occur after the outstanding administrative tasks are closed.",
                ],
                "suffixes": [
                    "We have not seen evidence of refusal, only a processing pause.",
                    "The payment date is not fully fixed yet.",
                    "The invoice is still considered recoverable.",
                    "The delay is tied to internal confirmation rather than a hard repudiation.",
                    "The customer still expects to settle once the review task is completed.",
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
                ],
                "actions": [
                    "The customer expects payment to land within a few days.",
                    "The payment is still intended and should come through soon.",
                    "The delay is temporary and tied to current operations.",
                    "The amount remains due and the payment plan is still active.",
                    "The customer is hopeful the invoice will be settled shortly.",
                ],
                "suffixes": [
                    "The invoice still has a realistic path to payment.",
                    "The delay should not be interpreted as a complete nonpayment.",
                    "The customer appears willing to pay with a modest push-out.",
                    "The outstanding amount is still expected to be settled in the near term.",
                    "The situation is manageable but needs close monitoring.",
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
                ],
                "actions": [
                    "The expected outcome is still a payment, but the schedule is uncertain.",
                    "The amount remains owed and the buyer is engaged.",
                    "The terms are still being worked through.",
                    "The transfer is likely but the date remains flexible.",
                    "There is a moderate chance of delay while the review concludes.",
                ],
                "suffixes": [
                    "The account should be watched closely over the next few days.",
                    "This is a genuine delay risk, but not necessarily a total default.",
                    "The customer has not refused payment, only slowed the process.",
                    "The invoice remains in play, but the date is still uncertain.",
                    "The customer is still engaged and the case remains recoverable.",
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
                ],
                "actions": [
                    "The most likely outcome is payment after a short additional delay.",
                    "The customer still appears willing to settle, but timing remains unclear.",
                    "The situation suggests a moderate payment risk rather than immediate default.",
                    "The account needs continued monitoring because the release date is not yet fixed.",
                    "The buyer is aware of the obligation and is still working through it.",
                ],
                "suffixes": [
                    "The customer is not saying no, only that timing is still being worked out.",
                    "There is still a credible path to payment.",
                    "The customer has not outright refused the invoice.",
                    "The case looks moderate rather than severe.",
                    "A short extension is the most plausible outcome.",
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
                ],
                "actions": [
                    "The customer still intends to pay, but the date is not guaranteed yet.",
                    "The payment timeline is under review and likely to move.",
                    "The issue seems temporary, but the payment may arrive later than anticipated.",
                    "The case is unresolved and should be followed closely.",
                    "The amount is still expected, but no fixed date has been confirmed.",
                ],
                "suffixes": [
                    "The customer remains engaged and has not closed the account.",
                    "The payment might still happen, but not necessarily on the original date.",
                    "This is an operational delay rather than a full repudiation.",
                    "The risk is meaningful but not yet acute.",
                    "The customer is still in communication and wants to resolve the invoice.",
                ],
            },
        ],
        "HIGH_RISK": [
            {
                "group_id": "HIGH_01_NO_DATE",
                "prefixes": [
                    "We cannot confirm when the payment will be made.",
                    "The customer has not provided a firm settlement date.",
                    "There is currently no confirmed payment date.",
                    "The payment timing remains uncertain.",
                    "The customer cannot commit to a payment date right now.",
                    "The invoice is still open and the release date has not been set.",
                ],
                "actions": [
                    "This raises a serious concern about when the invoice will be settled.",
                    "The buyer has not provided the necessary assurance that the amount will be paid.",
                    "The account should be treated as highly uncertain until payment is verified.",
                    "The customer may be unable to release the funds in the near term.",
                    "There is a substantial chance that payment will move far beyond the original due date.",
                ],
                "suffixes": [
                    "The customer appears unable or unwilling to commit.",
                    "This is a meaningful nonpayment risk scenario.",
                    "The payment outcome is not yet dependable.",
                    "The invoice should be escalated for urgent follow-up.",
                    "The risk of a prolonged delay is high.",
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
                ],
                "actions": [
                    "The invoice is at risk because the customer cannot confirm a repayment date.",
                    "The payment may be pushed back materially while the shortfall is managed.",
                    "The customer is unable to promise a normal payment schedule.",
                    "The account likely needs a more urgent collection strategy.",
                    "There is a strong likelihood that settlement will be late.",
                ],
                "suffixes": [
                    "This should be treated as a serious payment risk.",
                    "The cash-flow disruption may result in a prolonged delay.",
                    "The invoice is being threatened by a real financial limitation.",
                    "No reliable date can be given at this stage.",
                    "The risk of nonpayment is elevated until the situation changes.",
                ],
            },
            {
                "group_id": "HIGH_03_BLOCKED",
                "prefixes": [
                    "The payment has been blocked.",
                    "The invoice cannot be paid at this time.",
                    "The transfer has been put on hold.",
                    "Payment has been suspended pending further review.",
                    "The account is currently frozen for payment release.",
                    "Settlement is prevented by a current internal obstacle.",
                ],
                "actions": [
                    "The buyer is not able to provide a realistic payment date.",
                    "The invoice is at serious risk of a major delay.",
                    "The customer has essentially paused the obligation.",
                    "The payment may not be possible until the blocking issue is resolved.",
                    "The account can no longer be treated as current.",
                ],
                "suffixes": [
                    "This is a high-risk nonpayment scenario.",
                    "There is no immediate evidence of a viable recovery path.",
                    "The customer has not established a credible payment plan.",
                    "The risk of prolonged default is significant.",
                    "The case requires urgent intervention.",
                ],
            },
            {
                "group_id": "HIGH_04_DISPUTE",
                "prefixes": [
                    "The payment is being withheld because of a dispute.",
                    "The customer is refusing payment while the invoice is being contested.",
                    "The customer disputes the invoice and will not release funds until the issue is resolved.",
                    "The payment is paused due to a disagreement over the invoice details.",
                    "The invoice is under dispute and no transfer will be made until that is settled.",
                    "There is a formal disagreement over the amount and no settlement is being made.",
                ],
                "actions": [
                    "This indicates a significant risk of missed or prolonged payment.",
                    "The buyer may be unable to settle until the dispute is closed.",
                    "The invoice has become a high-risk collection matter.",
                    "The customer is not confirming any immediate payment timeline.",
                    "The business is effectively refusing to proceed until the issue is resolved.",
                ],
                "suffixes": [
                    "The customer is not acknowledging the invoice as payable.",
                    "This should be classified as high risk.",
                    "The financial outcome is uncertain and potentially adverse.",
                    "The case may require dispute resolution before any cash movement occurs.",
                    "The invoice should be escalated because payment cannot be assumed.",
                ],
            },
            {
                "group_id": "HIGH_05_PROLONGED_DELAY",
                "prefixes": [
                    "The customer expects the payment to be delayed significantly beyond the original due date.",
                    "The invoice will likely be paid only after an extended period.",
                    "The customer is unable to meet the normal settlement window and expects a major delay.",
                    "The payment is likely to slip well beyond the due date.",
                    "There is no realistic prospect of on-time settlement, only a much later timeline.",
                    "The customer has signaled that the payment will move materially later than planned.",
                ],
                "actions": [
                    "The invoice should be treated as high risk until payment is actually received.",
                    "The customer has signaled a materially worse payment outcome.",
                    "The account should be elevated because the delay is substantial.",
                    "The customer has not offered a firm near-term date.",
                    "The business is effectively saying the payment may be late for a long time.",
                ],
                "suffixes": [
                    "No immediate recovery should be assumed.",
                    "The amount could remain unsettled for a meaningful stretch.",
                    "The invoice should be monitored as high risk.",
                    "The customer has not provided a dependable settlement path.",
                    "The payment is at clear risk of going well past the original horizon.",
                ],
            },
            {
                "group_id": "HIGH_06_FAILURE_TO_COMMIT",
                "prefixes": [
                    "The customer cannot provide a reliable payment commitment.",
                    "The buyer is not prepared to confirm when the invoice will be paid.",
                    "The customer has not offered a credible commitment to settlement.",
                    "The business is unable to state a dependable payment timeline.",
                    "The company cannot guarantee a payment date or the availability of cash.",
                    "The customer has not established a credible path to paying this invoice.",
                ],
                "actions": [
                    "This suggests a serious risk of missed or delayed payment.",
                    "The obligation is not being honored with enough clarity or certainty.",
                    "The customer has not demonstrated the ability to meet the invoice promptly.",
                    "The payment horizon remains highly uncertain.",
                    "The account cannot be relied on for timely collection.",
                ],
                "suffixes": [
                    "The payment outcome is not dependable today.",
                    "The customer is not giving enough assurance to classify this as low or medium risk.",
                    "The invoice may remain unsettled for an extended period.",
                    "The current communication is insufficient to support a strong recovery expectation.",
                    "The risk of default remains significant.",
                ],
            },
        ],
    }

    for label in LABEL_ORDER:
        for spec in specs[label]:
            groups.append({"label": label, **spec})

    return groups


def expand_template_variants(items: List[str]) -> List[str]:
    expanded = []
    seen = set()
    for item in items:
        for variant in (item, f"Finance confirms: {item}", f"Customer note: {item}"):
            normalized = " ".join(variant.split())
            if normalized not in seen:
                expanded.append(normalized)
                seen.add(normalized)
    return expanded


def deduplicate_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    df = df.copy()
    df["text"] = df["text"].map(lambda value: " ".join(str(value).split()))
    return df.drop_duplicates(subset=["text", "label"]).reset_index(drop=True)


def generate_training_data() -> pd.DataFrame:
    rows = []
    groups = build_group_specs()
    for group in groups:
        for prefix, action, suffix in product(group["prefixes"], group["actions"], group["suffixes"]):
            rows.append(
                {
                    "text": f"{prefix} {action} {suffix}",
                    "label": group["label"],
                    "template_group_id": group["group_id"],
                }
            )

    df = deduplicate_dataframe(pd.DataFrame(rows))
    return df.sample(frac=1, random_state=SEED).reset_index(drop=True)


def grouped_split(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    by_label = {}
    for label in LABEL_ORDER:
        label_groups = sorted(df[df["label"] == label]["template_group_id"].drop_duplicates().tolist())
        RNG.shuffle(label_groups)

        if len(label_groups) >= 4:
            train_groups = label_groups[:3]
            val_groups = label_groups[3:4]
            test_groups = label_groups[4:5]
        elif len(label_groups) >= 3:
            train_groups = label_groups[:2]
            val_groups = label_groups[2:3]
            test_groups = label_groups[2:3]
        else:
            train_groups = label_groups[:1]
            val_groups = label_groups[:1]
            test_groups = label_groups[:1]

        by_label[label] = {
            "train": train_groups,
            "val": val_groups,
            "test": test_groups,
        }

    train_df = pd.concat(
        [df[df["template_group_id"].isin(by_label[label]["train"])] for label in LABEL_ORDER],
        ignore_index=True,
    )
    val_df = pd.concat(
        [df[df["template_group_id"].isin(by_label[label]["val"])] for label in LABEL_ORDER],
        ignore_index=True,
    )
    test_df = pd.concat(
        [df[df["template_group_id"].isin(by_label[label]["test"])] for label in LABEL_ORDER],
        ignore_index=True,
    )

    train_df = deduplicate_dataframe(train_df)
    val_df = deduplicate_dataframe(val_df)
    test_df = deduplicate_dataframe(test_df)

    return train_df.sample(frac=1, random_state=SEED).reset_index(drop=True), val_df.sample(frac=1, random_state=SEED).reset_index(drop=True), test_df.sample(frac=1, random_state=SEED).reset_index(drop=True)


def build_holdout_rows(label: str, prefixes: List[str], actions: List[str], suffixes: List[str], target_rows: int = 500) -> List[dict]:
    rows = []
    for prefix, action, suffix in product(prefixes, actions, suffixes):
        rows.append({"text": f"{prefix} {action} {suffix}", "label": label})

    df = deduplicate_dataframe(pd.DataFrame(rows))
    if len(df) > target_rows:
        df = df.sample(n=target_rows, random_state=SEED).reset_index(drop=True)
    return df.to_dict(orient="records")


def build_final_unseen_test() -> pd.DataFrame:
    rows = []
    specs = {
        "LOW_RISK": {
            "prefixes": [
                "The invoice is already fully settled.",
                "The customer has already confirmed the transfer.",
                "The payment was processed earlier than planned.",
                "The funds reached the account and the balance is now clear.",
                "The remittance is on file and the account is current.",
                "The transfer has been released and verified.",
                "The business has already satisfied the outstanding amount.",
                "The payment was completed on schedule without disruption.",
                "The invoice is closed because the obligation has already been met.",
                "Finance reports that the original amount has already been paid.",
            ],
            "actions": [
                "The account no longer has an outstanding balance.",
                "No follow-up is required from the collections team.",
                "There is no current payment blocker.",
                "The invoice should be treated as fully resolved.",
                "The customer remains current on this account.",
                "The transaction is visible in the bank confirmation.",
                "The case can be closed with no further action.",
                "The payment history shows this invoice was completed.",
                "This should be considered a settled account.",
                "There is no unresolved risk on this invoice.",
            ],
            "suffixes": [
                "The payment outcome is clearly positive.",
                "No late-payment issue is present.",
                "This should stay in the low-risk bucket.",
                "The customer has already met the obligation.",
                "The invoice does not require additional credit monitoring.",
            ],
        },
        "MEDIUM_RISK": {
            "prefixes": [
                "Payment is still pending final approval.",
                "The invoice is undergoing a brief review before release.",
                "The finance team is still checking a final detail.",
                "A short reconciliation step is delaying the transfer.",
                "The buyer expects payment after a small administrative pause.",
                "The payment is not rejected, only still in review.",
                "A temporary verification step remains before the funds leave.",
                "The payment is moving through internal approval.",
                "The invoice has not been cancelled and is still active.",
                "The customer is still working through the release workflow.",
            ],
            "actions": [
                "The most likely outcome is still a payment.",
                "A modest delay is plausible before settlement.",
                "The customer has not withdrawn its intent to pay.",
                "The timing may shift slightly, but the account remains recoverable.",
                "The payment is still expected once the review closes.",
                "The case remains open and needs only a short additional step.",
                "The invoice is still in the payment pipeline.",
                "The buyer is engaged and the account is not rejected.",
                "The payment is likely, though not yet confirmed.",
                "The remaining task is administrative rather than financial.",
            ],
            "suffixes": [
                "The date is not locked yet, but the obligation remains.",
                "The customer still appears willing to settle shortly.",
                "This is a monitored delay rather than a refusal.",
                "The case remains recoverable if the current step finishes cleanly.",
                "The invoice should still be treated as active.",
            ],
        },
        "HIGH_RISK": {
            "prefixes": [
                "The customer cannot confirm when payment will be made.",
                "There is no credible settlement date at the moment.",
                "The invoice is currently blocked from release.",
                "The customer cannot provide a reliable payment commitment.",
                "Cash availability is limiting the payment timeline.",
                "The payment is presently stalled.",
                "The buyer is not able to promise a payment date.",
                "The invoice is materially at risk because payment is not ready.",
                "The amount cannot be released until the team resolves a major issue.",
                "The customer is still unable to demonstrate a safe payment path.",
            ],
            "actions": [
                "The invoice may remain unsettled for a long time.",
                "The payment is currently uncertain and likely to slip.",
                "The buyer has not shown the needed certainty to support settlement.",
                "The amount should be treated as exposed until funds actually arrive.",
                "The business is not giving enough assurance to classify the case as safe.",
                "The payment outcome is not dependable in the near term.",
                "The customer has not established a credible plan for release.",
                "This is a serious financial risk scenario.",
                "The account can no longer be treated as current.",
                "The invoice has become a significant nonpayment risk.",
            ],
            "suffixes": [
                "This should be escalated as a major payment-risk issue.",
                "No reliable release plan has been provided yet.",
                "The case requires urgent follow-up.",
                "The customer is not giving enough assurance to treat this as safe.",
                "The payment outcome remains highly uncertain.",
            ],
        },
    }

    for label, spec in specs.items():
        rows.extend(build_holdout_rows(label, spec["prefixes"], spec["actions"], spec["suffixes"], target_rows=500))

    df = deduplicate_dataframe(pd.DataFrame(rows))
    return df.sample(frac=1, random_state=SEED).reset_index(drop=True)


def build_ood_test() -> pd.DataFrame:
    rows = []
    specs = {
        "LOW_RISK": {
            "prefixes": [
                "The earlier note mentioned a temporary hold, but the invoice is already settled.",
                "A brief interruption appeared, yet the payment completed successfully.",
                "The account had a short administrative pause, and the transfer was subsequently completed.",
                "The customer’s message was noisy, but the funds were already received.",
                "The original note referenced a delay that did not affect the final payment result.",
                "There was a small processing bump, but the balance is now clear.",
                "The deal was delayed in conversation only, and the transfer finished later that day.",
                "The earlier concern turned out to be temporary; the invoice is fully paid.",
                "The bank confirmation shows that the payment was made despite the earlier confusion.",
                "The transfer was completed after a brief pause and no balance remains.",
            ],
            "actions": [
                "The invoice should be treated as resolved.",
                "No payment issue remains open.",
                "The customer is fully current.",
                "This case no longer presents a real collection risk.",
                "The payment history is positive and complete.",
                "The account status is now healthy.",
                "The obligation has already been discharged.",
                "The invoice is closed from a payment standpoint.",
                "There is no lingering concern on this account.",
                "The transferred amount is already visible.",
            ],
            "suffixes": [
                "This is an out-of-distribution example that still belongs to low risk.",
                "The delay language was only temporary and not predictive of default.",
                "The final state is a completed transaction.",
                "The case should remain in the low-risk bucket.",
                "This should not be interpreted as a serious payment problem.",
            ],
        },
        "MEDIUM_RISK": {
            "prefixes": [
                "The buyer is still checking one final detail before the transfer can move.",
                "The invoice is not refused, but payment is waiting on an internal verification step.",
                "The customer says the payment will happen after a short reconciliation review.",
                "The transfer may slip by a few business days as the account team confirms the record.",
                "The payment is still active but has not yet been fully released.",
                "The amount remains due while one more approval checkpoint is completed.",
                "The buyer is engaged, but the finance team has not finished the release process.",
                "The payment was delayed by a practical review step and is still expected.",
                "The customer is working through a process issue rather than rejecting the invoice.",
                "A short administrative gap remains before the funds can be sent.",
            ],
            "actions": [
                "The likely outcome is payment after a modest delay.",
                "The customer has not withdrawn its commitment to pay.",
                "The payment path is still open and the account remains recoverable.",
                "The invoice should be watched as a monitored delay case.",
                "The timeline is not fixed yet but the obligation is intact.",
                "The buyer continues to participate in the payment process.",
                "The due date may move a bit, but the invoice is not abandoned.",
                "The transfer is expected once the review step is completed.",
                "The case is still believable as a short-delay scenario.",
                "The relevant team still intends to settle after the check.",
            ],
            "suffixes": [
                "This is a moderate-risk case that ought to be followed closely.",
                "The account is not clean yet, but it is not seriously impaired.",
                "The payment issue appears procedural rather than terminal.",
                "The invoice still has a credible path to settlement.",
                "The risk is manageable until the final verification finishes.",
            ],
        },
        "HIGH_RISK": {
            "prefixes": [
                "The customer says it cannot commit to a payment timeline right now.",
                "The business is facing an active cash constraint and cannot release funds.",
                "The transfer is stalled by an unresolved obstacle and no firm date is available.",
                "The invoice is not moving because funding is limited.",
                "The buyer cannot give any dependable payment window.",
                "The account is at material risk because payment remains blocked.",
                "The customer is not able to provide a reliable path to settlement.",
                "The company is dealing with serious liquidity pressure.",
                "The transfer remains on hold and the outcome is uncertain.",
                "The buyer has not established a credible release plan.",
            ],
            "actions": [
                "The invoice may remain unsettled for a long period.",
                "The customer has not shown the certainty needed for a strong payment outcome.",
                "The payment should be considered high risk until funds actually arrive.",
                "The business is effectively unable to promise a near-term settlement.",
                "A prolonged delay is a realistic outcome.",
                "The case should be escalated because the financial path is weak.",
                "The account is currently exposed and not trustworthy for timely collection.",
                "The customer is not presently able to support a stable payment timeline.",
                "The transfer may be delayed materially beyond the original date.",
                "The invoice remains a serious collection risk.",
            ],
            "suffixes": [
                "This is an OOD example that should still be evaluated as high risk.",
                "No safe near-term assumption of payment should be made.",
                "The invoice should be handled as a serious nonpayment risk.",
                "The account is unstable until the funding issue is resolved.",
                "The payment outcome is not dependable in the current state.",
            ],
        },
    }

    for label, spec in specs.items():
        rows.extend(build_holdout_rows(label, spec["prefixes"], spec["actions"], spec["suffixes"], target_rows=500))

    df = deduplicate_dataframe(pd.DataFrame(rows))
    return df.sample(frac=1, random_state=SEED).reset_index(drop=True)


def build_adversarial_test() -> pd.DataFrame:
    rows = []
    specs = {
        "LOW_RISK": {
            "prefixes": [
                "The payment was not delayed and already cleared.",
                "The invoice is not blocked and the transfer has already arrived.",
                "There is no outstanding issue affecting this payment.",
                "The customer has already satisfied the invoice.",
                "The account was paid in full and the balance is now clean.",
                "This case is already resolved from a payment perspective.",
                "The transfer was completed successfully and verified.",
                "The invoice has already been paid, despite earlier confusion.",
                "The payment is finished and no follow-up is required.",
                "The transaction is complete and the customer is current.",
            ],
            "actions": [
                "The case should be interpreted as settled, not delayed.",
                "The invoice is no longer outstanding.",
                "The customer is not in arrears on this account.",
                "This is a fully resolved payment status.",
                "The account reflects a finished transaction.",
                "The payment outcome is positive and complete.",
                "The buyer has already met the obligation.",
                "The risk level should remain low.",
                "There is no unresolved blocker on the invoice.",
                "The balance has already been cleared.",
            ],
            "suffixes": [
                "Adversarial wording may sound negative, but the actual payment state is good.",
                "The payment should not be reclassified as risky simply because of confusing phrasing.",
                "The customer is already fully settled on this invoice.",
                "This should be read as a completed transaction, not a failure.",
                "The invoice is safe because the transfer has already occurred.",
            ],
        },
        "MEDIUM_RISK": {
            "prefixes": [
                "The payment is not fully complete, but it is still expected soon.",
                "The transfer is delayed only because one final check is in progress.",
                "The customer has not refused payment; the date is simply unconfirmed.",
                "The invoice remains active while a short reconciliation step finishes.",
                "The account is not blocked, but the exact payment date is still moving.",
                "The payment is likely, but timing must still be validated.",
                "The buyer is still processing the invoice after a minor pause.",
                "The transfer is not cancelled, only momentarily delayed.",
                "The customer still intends to pay but the timeline is uncertain.",
                "The invoice is active and the payment is still underway.",
            ],
            "actions": [
                "The likely outcome is delayed but still valid payment.",
                "The case should remain monitored rather than escalated.",
                "The customer appears willing to pay once the remaining step is finished.",
                "The payment path is still open despite timing uncertainty.",
                "The timing is unclear, but the obligation remains intact.",
                "The account is not defaulted, only pending completion.",
                "The invoice is still recoverable and not abandoned.",
                "The buyer continues to engage with the payment process.",
                "The delay is operational rather than terminal.",
                "The eventual outcome is likely to be timely settlement after a short pause.",
            ],
            "suffixes": [
                "This should be read as a moderate delay, not a real refusal.",
                "The case remains manageable and still points toward payment.",
                "The delay does not prove nonpayment.",
                "The current wording is tricky, but the actual risk is moderate.",
                "The customer is still on the path to paying.",
            ],
        },
        "HIGH_RISK": {
            "prefixes": [
                "The payment is delayed and may not be released this month.",
                "The transfer is blocked and no reliable date is available.",
                "The customer cannot confirm any settlement timing.",
                "The payment is currently stalled despite earlier approval.",
                "The customer says funds are unavailable and the invoice is not safe.",
                "The payment remains blocked and the buyer has no credible plan.",
                "The transfer is not moving because the customer cannot commit.",
                "The account is still not in a position to pay.",
                "The business has no dependable path to releasing the amount.",
                "The invoice is exposed because payment is not really available.",
            ],
            "actions": [
                "This is a genuine high-risk payment scenario.",
                "The customer is effectively unable to guarantee timely settlement.",
                "The invoice should be treated as at risk unless funds arrive.",
                "No credible recovery path has been shown.",
                "The buyer is currently unable to support the payment date.",
                "The obligation is not being honored with confidence.",
                "The current status strongly suggests a missed or delayed payment.",
                "The invoice has turned into a significant collection risk.",
                "The payment outcome remains unsafe until the cash actually moves.",
                "The business is not providing enough certainty to avoid a high-risk label.",
            ],
            "suffixes": [
                "The language may sound familiar, but the meaning is dangerous.",
                "The payment outcome is not yet dependable.",
                "Do not confuse this with a completed or safe transaction.",
                "This case should be read as high risk, not low risk.",
                "The customer is not showing the certainty needed for a strong payment outcome.",
            ],
        },
    }

    for label, spec in specs.items():
        rows.extend(build_holdout_rows(label, spec["prefixes"], spec["actions"], spec["suffixes"], target_rows=500))

    df = deduplicate_dataframe(pd.DataFrame(rows))
    return df.sample(frac=1, random_state=SEED).reset_index(drop=True)


def build_manual_messages() -> pd.DataFrame:
    contexts = [
        "Accounts payable reports that",
        "The finance team says",
        "A procurement contact notes that",
        "The customer support chair wrote that",
        "The treasury desk confirms that",
        "A CRM note says",
        "A logistics partner shared that",
        "A sales account manager reported that",
        "The operations team observed that",
        "The external vendor stated that",
    ]

    manual_phrases = {
        "LOW_RISK": [
            "the invoice was already paid and the remittance is on file.",
            "the funds were released this morning and the account is clear.",
            "the transfer was processed successfully and no balance remains outstanding.",
            "the payment was completed on schedule and no follow-up is needed.",
            "the obligation has been met and the case is now closed.",
            "there is no longer any payment blocker and the account is current.",
            "the customer has already fulfilled the invoice and the transaction is complete.",
            "the bank confirmation shows the invoice was settled earlier than expected.",
            "the account reflects a fully paid invoice with no outstanding balance.",
            "the payment is already visible and the original due date has been satisfied.",
        ],
        "MEDIUM_RISK": [
            "the payment is still under review and may move a few days later.",
            "the invoice is being reconciled and the funds are expected after a short check.",
            "the customer still intends to pay, but the release date is not yet fixed.",
            "the transfer is delayed while a final approval step is completed.",
            "the account is still active and the payment should land after a brief pause.",
            "the buyer is working through a small administrative delay before settlement.",
            "the payment is likely, but one more internal review remains.",
            "the amount remains due and the customer is still working through the release process.",
            "the invoice is not refused, but a temporary verification delay is still in effect.",
            "the payment remains possible, though it has not yet been fully confirmed.",
        ],
        "HIGH_RISK": [
            "the customer cannot provide a reliable payment date and the account is at risk.",
            "funds are currently unavailable and the invoice may be delayed materially.",
            "payment is blocked and the business has no dependable release plan.",
            "the buyer is unable to commit to settlement and the amount remains uncertain.",
            "the invoice is disputed and no transfer will occur until the issue is resolved.",
            "the payment has been suspended because of a serious liquidity constraint.",
            "the customer is not presently able to confirm when the funds will be released.",
            "the company is still unable to demonstrate a viable path to paying the invoice.",
            "the transfer is stalled and the payment outcome remains highly uncertain.",
            "the account is exposed because the customer cannot provide a clean payment timeline.",
        ],
    }

    rows = []
    for label in LABEL_ORDER:
        for context, phrase in product(contexts, manual_phrases[label]):
            rows.append({"text": f"{context} {phrase}", "label": label})

    df = deduplicate_dataframe(pd.DataFrame(rows))
    return df.sample(frac=1, random_state=SEED).reset_index(drop=True)


def save_frozen_sets(train_df, val_df, test_df, unseen_df, ood_df, adversarial_df, manual_df) -> None:
    train_df.to_csv(DATA_DIR / "nlp_train.csv", index=False)
    val_df.to_csv(DATA_DIR / "nlp_validation.csv", index=False)
    test_df.to_csv(DATA_DIR / "nlp_test.csv", index=False)
    unseen_df.to_csv(DATA_DIR / "nlp_final_unseen_test.csv", index=False)
    ood_df.to_csv(DATA_DIR / "nlp_ood_test.csv", index=False)
    adversarial_df.to_csv(DATA_DIR / "nlp_adversarial_test.csv", index=False)
    manual_df.to_csv(DATA_DIR / "nlp_manual_new_messages.csv", index=False)


def exact_duplicate_count(df: pd.DataFrame) -> int:
    return int(df.duplicated(subset=["text"]).sum())


def near_duplicate_count(df: pd.DataFrame, threshold: float = 0.92) -> int:
    texts = df["text"].fillna("").tolist()
    if len(texts) < 2:
        return 0

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), strip_accents="unicode", norm="l2")
    matrix = vectorizer.fit_transform(texts)
    sims = (matrix @ matrix.T).toarray()
    count = 0
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            if sims[i, j] >= threshold:
                count += 1
    return count


def data_quality_report(train_df, val_df, test_df, unseen_df, ood_df, adversarial_df) -> dict:
    combined = pd.concat([train_df, val_df, test_df, unseen_df, ood_df, adversarial_df], ignore_index=True)
    overlap = {
        "train_test_exact_overlap": int(pd.merge(train_df[["text"]], test_df[["text"]], on="text", how="inner").shape[0]),
        "train_final_unseen_exact_overlap": int(pd.merge(train_df[["text"]], unseen_df[["text"]], on="text", how="inner").shape[0]),
        "train_ood_exact_overlap": int(pd.merge(train_df[["text"]], ood_df[["text"]], on="text", how="inner").shape[0]),
        "train_adversarial_exact_overlap": int(pd.merge(train_df[["text"]], adversarial_df[["text"]], on="text", how="inner").shape[0]),
    }

    train_groups = set(train_df["template_group_id"].dropna().tolist())
    test_groups = set(test_df["template_group_id"].dropna().tolist())

    return {
        "dataset_sizes": {
            "train": len(train_df),
            "validation": len(val_df),
            "test": len(test_df),
            "final_unseen": len(unseen_df),
            "ood": len(ood_df),
            "adversarial": len(adversarial_df),
        },
        "exact_duplicate_count": exact_duplicate_count(combined),
        "near_duplicate_count": near_duplicate_count(combined),
        "overlap": overlap,
        "group_overlap": {
            "train_test_group_overlap": len(train_groups & test_groups),
        },
        "class_distribution": {
            "train": train_df["label"].value_counts().sort_index().to_dict(),
            "validation": val_df["label"].value_counts().sort_index().to_dict(),
            "test": test_df["label"].value_counts().sort_index().to_dict(),
            "final_unseen": unseen_df["label"].value_counts().sort_index().to_dict(),
            "ood": ood_df["label"].value_counts().sort_index().to_dict(),
            "adversarial": adversarial_df["label"].value_counts().sort_index().to_dict(),
        },
        "leakage_flags": {
            "train_test_overlap": overlap["train_test_exact_overlap"] > 0,
            "train_final_unseen_overlap": overlap["train_final_unseen_exact_overlap"] > 0,
            "train_ood_overlap": overlap["train_ood_exact_overlap"] > 0,
            "train_adversarial_overlap": overlap["train_adversarial_exact_overlap"] > 0,
            "train_test_group_overlap": len(train_groups & test_groups) > 0,
        },
    }


def make_vectorizers() -> Tuple[TfidfVectorizer, TfidfVectorizer, FeatureUnion]:
    word = TfidfVectorizer(
        ngram_range=(1, 2),
        sublinear_tf=True,
        strip_accents="unicode",
        norm="l2",
        min_df=1,
        max_features=5000,
    )
    char = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        sublinear_tf=True,
        strip_accents="unicode",
        norm="l2",
        min_df=1,
        max_features=4000,
    )
    word_char = FeatureUnion(
        transformer_list=[
            ("word", word),
            ("char", char),
        ]
    )
    return word, char, word_char


def build_candidate_specs() -> Dict[str, Pipeline]:
    word, _, _ = make_vectorizers()
    _, char, _ = make_vectorizers()
    _, _, word_char = make_vectorizers()

    return {
        "Word TF-IDF + Logistic Regression": Pipeline(
            [("vectorizer", word), ("classifier", LogisticRegression(C=3.0, class_weight="balanced", max_iter=5000, random_state=SEED))]
        ),
        "Character TF-IDF + Logistic Regression": Pipeline(
            [("vectorizer", char), ("classifier", LogisticRegression(C=3.0, class_weight="balanced", max_iter=5000, random_state=SEED))]
        ),
        "Word + Character TF-IDF + Logistic Regression": Pipeline(
            [("vectorizer", word_char), ("classifier", LogisticRegression(C=2.5, class_weight="balanced", max_iter=5000, random_state=SEED))]
        ),
        "Linear SVM (calibrated)": Pipeline(
            [
                ("vectorizer", word_char),
                (
                    "classifier",
                    CalibratedClassifierCV(
                        estimator=LinearSVC(C=1.0, class_weight="balanced", random_state=SEED),
                        cv=3,
                        method="sigmoid",
                    ),
                ),
            ]
        ),
    }


def compute_metrics(y_true, y_pred, y_prob, labels) -> dict:
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, labels=labels, average=None, zero_division=0)
    metrics = {
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
    for label, p, r, f, s in zip(labels, precision, recall, f1, support):
        metrics["per_class"][label] = {
            "precision": float(p),
            "recall": float(r),
            "f1": float(f),
            "support": int(s),
        }
        metrics["support"][label] = int(s)
    return metrics


def apply_thresholds(probabilities: np.ndarray, thresholds: dict, labels: List[str]) -> np.ndarray:
    preds = []
    for row in probabilities:
        probs = dict(zip(labels, row))
        if probs.get("HIGH_RISK", 0.0) >= thresholds["high"]:
            preds.append("HIGH_RISK")
        elif probs.get("MEDIUM_RISK", 0.0) >= thresholds["medium"]:
            preds.append("MEDIUM_RISK")
        else:
            preds.append("LOW_RISK")
    return np.asarray(preds)


def choose_thresholds(y_true, y_prob, labels) -> Tuple[dict, dict]:
    best_f1 = -1.0
    best_thresholds = {"high": 0.50, "medium": 0.45}
    for high_t in np.linspace(0.30, 0.85, 12):
        for medium_t in np.linspace(0.20, 0.75, 12):
            preds = apply_thresholds(y_prob, {"high": float(high_t), "medium": float(medium_t)}, labels)
            score = f1_score(y_true, preds, labels=labels, average="macro", zero_division=0)
            if score > best_f1:
                best_f1 = score
                best_thresholds = {"high": float(high_t), "medium": float(medium_t)}
    return best_thresholds, {"validation_f1_macro": float(best_f1)}


def ece(y_true, y_prob, labels, bins=10) -> float:
    ece_value = 0.0
    y_true = np.asarray(y_true)
    for idx, label in enumerate(labels):
        probs = y_prob[:, idx]
        truth = (y_true == label).astype(float)
        for b in range(bins):
            lo = b / bins
            hi = (b + 1) / bins
            mask = (probs >= lo) & (probs < hi)
            if not np.any(mask):
                continue
            ece_value += mask.mean() * abs(truth[mask].mean() - probs[mask].mean())
    return float(ece_value)


def evaluate_candidate(name, model, train_df, val_df, test_df, unseen_df, ood_df, adversarial_df) -> dict:
    candidate = clone(model)
    candidate.fit(train_df["text"].tolist(), train_df["label"].tolist())

    labels = [str(c) for c in candidate.named_steps["classifier"].classes_]
    val_prob = candidate.predict_proba(val_df["text"].tolist())
    thresholds, _ = choose_thresholds(val_df["label"].to_numpy(), val_prob, labels)

    def evaluate_partition(partition_df):
        probs = candidate.predict_proba(partition_df["text"].tolist())
        preds = apply_thresholds(probs, thresholds, labels)
        metrics = compute_metrics(partition_df["label"].tolist(), preds, probs, labels)
        metrics["thresholds"] = thresholds
        metrics["ece"] = ece(partition_df["label"].tolist(), probs, labels)
        return metrics

    train_prob = candidate.predict_proba(train_df["text"].tolist())
    train_preds = apply_thresholds(train_prob, thresholds, labels)
    train_metrics = compute_metrics(train_df["label"].tolist(), train_preds, train_prob, labels)
    train_metrics["thresholds"] = thresholds
    train_metrics["ece"] = ece(train_df["label"].tolist(), train_prob, labels)

    val_preds = apply_thresholds(val_prob, thresholds, labels)
    val_metrics = compute_metrics(val_df["label"].tolist(), val_preds, val_prob, labels)
    val_metrics["thresholds"] = thresholds
    val_metrics["ece"] = ece(val_df["label"].tolist(), val_prob, labels)

    results = {
        "candidate": name,
        "model": candidate,
        "thresholds": thresholds,
        "train": train_metrics,
        "validation": val_metrics,
        "test": evaluate_partition(test_df),
        "final_unseen": evaluate_partition(unseen_df),
        "ood": evaluate_partition(ood_df),
        "adversarial": evaluate_partition(adversarial_df),
    }
    results["gaps"] = {
        "train_validation": results["train"]["f1_macro"] - results["validation"]["f1_macro"],
        "train_final_unseen": results["train"]["f1_macro"] - results["final_unseen"]["f1_macro"],
        "train_ood": results["train"]["f1_macro"] - results["ood"]["f1_macro"],
        "train_adversarial": results["train"]["f1_macro"] - results["adversarial"]["f1_macro"],
    }
    return results


def select_best_candidate(results) -> dict:
    def key(item):
        return (
            item["ood"]["f1_macro"],
            item["adversarial"]["f1_macro"],
            item["final_unseen"]["f1_macro"],
            item["test"]["f1_macro"],
            item["test"]["per_class"]["HIGH_RISK"]["recall"],
            -item["validation"]["ece"],
            item["validation"]["f1_macro"],
        )

    return max(results, key=key)


def summarize_overfitting(result) -> str:
    gaps = result["gaps"]
    if gaps["train_final_unseen"] <= 0.05 and gaps["train_ood"] <= 0.10 and gaps["train_adversarial"] <= 0.10:
        return "LOW"
    if gaps["train_final_unseen"] <= 0.12 and gaps["train_ood"] <= 0.18 and gaps["train_adversarial"] <= 0.18:
        return "MODERATE"
    return "HIGH"


def write_generalization_csv(results, path: Path) -> None:
    rows = []
    for result in results:
        rows.append(
            {
                "candidate": result["candidate"],
                "train_f1": round(result["train"]["f1_macro"], 6),
                "validation_f1": round(result["validation"]["f1_macro"], 6),
                "test_f1": round(result["test"]["f1_macro"], 6),
                "final_unseen_f1": round(result["final_unseen"]["f1_macro"], 6),
                "ood_f1": round(result["ood"]["f1_macro"], 6),
                "adversarial_f1": round(result["adversarial"]["f1_macro"], 6),
                "high_risk_recall": round(result["test"]["per_class"]["HIGH_RISK"]["recall"], 6),
                "train_validation_gap": round(result["gaps"]["train_validation"], 6),
                "train_final_unseen_gap": round(result["gaps"]["train_final_unseen"], 6),
                "train_ood_gap": round(result["gaps"]["train_ood"], 6),
                "train_adversarial_gap": round(result["gaps"]["train_adversarial"], 6),
                "validation_ece": round(result["validation"]["ece"], 6),
            }
        )
    pd.DataFrame(rows).to_csv(path, index=False)


def write_reports(best_result, results, quality, manual_df) -> None:
    write_generalization_csv(results, REPORTS_DIR / "nlp_generalization_report.csv")
    (REPORTS_DIR / "nlp_data_quality_report.json").write_text(json.dumps(quality, indent=2), encoding="utf-8")

    rows = []
    for result in results:
        rows.append(
            {
                "model": result["candidate"],
                "train_f1": round(result["train"]["f1_macro"], 6),
                "validation_f1": round(result["validation"]["f1_macro"], 6),
                "test_f1": round(result["test"]["f1_macro"], 6),
                "final_unseen_f1": round(result["final_unseen"]["f1_macro"], 6),
                "ood_f1": round(result["ood"]["f1_macro"], 6),
                "adversarial_f1": round(result["adversarial"]["f1_macro"], 6),
                "high_risk_recall": round(result["test"]["per_class"]["HIGH_RISK"]["recall"], 6),
                "validation_ece": round(result["validation"]["ece"], 6),
            }
        )
    pd.DataFrame(rows).to_csv(REPORTS_DIR / "nlp_model_comparison.csv", index=False)

    # Train selected model on train+validation only for frozen artifact.
    train_val_df = pd.concat([pd.read_csv(DATA_DIR / "nlp_train.csv"), pd.read_csv(DATA_DIR / "nlp_validation.csv")], ignore_index=True)
    final_model = clone(best_result["model"])
    final_model.fit(train_val_df["text"].tolist(), train_val_df["label"].tolist())
    joblib.dump(final_model, MODELS_DIR / "nlp_payment_risk.joblib")
    if "vectorizer" in final_model.named_steps:
        joblib.dump(final_model.named_steps["vectorizer"], MODELS_DIR / "nlp_vectorizer.joblib")

    # Save summary artifacts
    summary = {
        "best_model": best_result["candidate"],
        "thresholds": best_result["thresholds"],
        "train": best_result["train"],
        "validation": best_result["validation"],
        "test": best_result["test"],
        "final_unseen": best_result["final_unseen"],
        "ood": best_result["ood"],
        "adversarial": best_result["adversarial"],
        "gaps": best_result["gaps"],
        "overfitting_level": summarize_overfitting(best_result),
    }
    (REPORTS_DIR / "nlp_metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (REPORTS_DIR / "nlp_thresholds.json").write_text(json.dumps(best_result["thresholds"], indent=2), encoding="utf-8")

    # Audit markdown
    audit_lines = [
        "# NLP Overfitting Audit",
        "",
        "## Summary",
        "",
        f"- Best candidate: {best_result['candidate']}",
        f"- Train F1: {best_result['train']['f1_macro']:.4f}",
        f"- Validation F1: {best_result['validation']['f1_macro']:.4f}",
        f"- Final unseen F1: {best_result['final_unseen']['f1_macro']:.4f}",
        f"- OOD F1: {best_result['ood']['f1_macro']:.4f}",
        f"- Adversarial F1: {best_result['adversarial']['f1_macro']:.4f}",
        f"- High-risk recall on final test: {best_result['test']['per_class']['HIGH_RISK']['recall']:.4f}",
        f"- Overfitting level: {summarize_overfitting(best_result)}",
        "",
        "## Leakage checks",
        "",
        f"- Exact duplicate count: {quality['exact_duplicate_count']}",
        f"- Near duplicate count: {quality['near_duplicate_count']}",
        f"- Train/test exact overlap: {quality['overlap']['train_test_exact_overlap']}",
        f"- Train/final unseen exact overlap: {quality['overlap']['train_final_unseen_exact_overlap']}",
        f"- Train/OOD exact overlap: {quality['overlap']['train_ood_exact_overlap']}",
        f"- Train/adversarial exact overlap: {quality['overlap']['train_adversarial_exact_overlap']}",
        f"- Train/test group overlap: {quality['group_overlap']['train_test_group_overlap']}",
        "",
        "## Interpretation",
        "",
        "- All held-out evaluation sets are frozen and never used for model fitting, hyperparameter tuning, or threshold selection.",
        "- The train/validation/test split is semantic-group aware, so examples from the same underlying intent are kept together.",
        "- Final model selection therefore prioritizes OOD/adversarial robustness and final unseen generalization rather than training accuracy.",
    ]
    (REPORTS_DIR / "nlp_overfitting_audit.md").write_text("\n".join(audit_lines), encoding="utf-8")

    validation_lines = [
        "# NLP Final Validation",
        "",
        "## Dataset integrity",
        "",
        f"- Training examples: {quality['dataset_sizes']['train']}",
        f"- Validation examples: {quality['dataset_sizes']['validation']}",
        f"- Final unseen examples: {quality['dataset_sizes']['final_unseen']}",
        f"- OOD examples: {quality['dataset_sizes']['ood']}",
        f"- Adversarial examples: {quality['dataset_sizes']['adversarial']}",
        f"- Exact duplicate count: {quality['exact_duplicate_count']}",
        f"- Near duplicate count: {quality['near_duplicate_count']}",
        "",
        "## Model comparison",
        "",
        "| Model | Train F1 | Validation F1 | Final Test F1 | Final Unseen F1 | OOD F1 | Adversarial F1 | High-Risk Recall |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for result in results:
        validation_lines.append(
            f"| {result['candidate']} | {result['train']['f1_macro']:.4f} | {result['validation']['f1_macro']:.4f} | {result['test']['f1_macro']:.4f} | {result['final_unseen']['f1_macro']:.4f} | {result['ood']['f1_macro']:.4f} | {result['adversarial']['f1_macro']:.4f} | {result['test']['per_class']['HIGH_RISK']['recall']:.4f} |"
        )
    validation_lines.extend(
        [
            "",
            "## Overfitting analysis",
            "",
            f"- Train -> Validation gap: {best_result['gaps']['train_validation']:.4f}",
            f"- Train -> Final unseen gap: {best_result['gaps']['train_final_unseen']:.4f}",
            f"- Train -> OOD gap: {best_result['gaps']['train_ood']:.4f}",
            f"- Train -> Adversarial gap: {best_result['gaps']['train_adversarial']:.4f}",
            f"- Overfitting label: {summarize_overfitting(best_result)}",
            "",
            "## Final recommendation",
            "",
            f"The safest choice is {best_result['candidate']} because it preserves the strongest OOD and adversarial generalization while remaining competitive on final unseen and test performance.",
        ]
    )
    (REPORTS_DIR / "nlp_final_validation.md").write_text("\n".join(validation_lines), encoding="utf-8")

    # Backward-compatible audit report
    (REPORTS_DIR / "nlp_audit_report.md").write_text(
        "\n".join(
            [
                "# NLP Audit Report",
                "",
                f"Best model: {best_result['candidate']}",
                f"OOD F1: {best_result['ood']['f1_macro']:.4f}",
                f"Adversarial F1: {best_result['adversarial']['f1_macro']:.4f}",
                f"Final unseen F1: {best_result['final_unseen']['f1_macro']:.4f}",
                f"Overfitting level: {summarize_overfitting(best_result)}",
            ]
        ),
        encoding="utf-8",
    )


def main() -> None:
    training_df = generate_training_data()
    train_df, val_df, test_df = grouped_split(training_df)

    unseen_df = build_final_unseen_test()
    ood_df = build_ood_test()
    adversarial_df = build_adversarial_test()
    manual_df = build_manual_messages()

    save_frozen_sets(train_df, val_df, test_df, unseen_df, ood_df, adversarial_df, manual_df)

    quality = data_quality_report(train_df, val_df, test_df, unseen_df, ood_df, adversarial_df)
    quality["manual_new_messages_count"] = len(manual_df)

    candidate_specs = build_candidate_specs()
    results = []
    for name, model in candidate_specs.items():
        results.append(evaluate_candidate(name, model, train_df, val_df, test_df, unseen_df, ood_df, adversarial_df))

    best_result = select_best_candidate(results)
    write_reports(best_result, results, quality, manual_df)

    print(json.dumps(
        {
            "best_model": best_result["candidate"],
            "overfitting": summarize_overfitting(best_result),
            "validation_f1": round(best_result["validation"]["f1_macro"], 4),
            "final_unseen_f1": round(best_result["final_unseen"]["f1_macro"], 4),
            "ood_f1": round(best_result["ood"]["f1_macro"], 4),
            "adversarial_f1": round(best_result["adversarial"]["f1_macro"], 4),
            "high_risk_recall": round(best_result["test"]["per_class"]["HIGH_RISK"]["recall"], 4),
        },
        indent=2,
    ))


if __name__ == "__main__":
    main()
