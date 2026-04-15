import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from futures_pnl.contracts import upcoming_contract_schedule


class ContractScheduleTests(unittest.TestCase):
    def test_upcoming_contract_schedule_returns_next_five_future_expiries(self) -> None:
        schedule = upcoming_contract_schedule("2026-04-14", contract_count=5)

        self.assertEqual(len(schedule), 5)
        self.assertEqual(schedule[0]["expiry_date"], "2026-06-26")
        self.assertEqual(schedule[0]["contract_code"], "FMASI20JUI26")
        self.assertTrue(all(item["days_to_expiry"] > 0 for item in schedule))
        self.assertEqual(
            [item["contract_code"] for item in schedule[:4]],
            ["FMASI20JUI26", "FMASI20SEP26", "FMASI20DEC26", "FMASI20MAR27"],
        )
        self.assertEqual(schedule[-1]["contract_code"], "FMASI20JUI27")


if __name__ == "__main__":
    unittest.main()
