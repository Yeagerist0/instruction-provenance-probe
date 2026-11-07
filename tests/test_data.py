import unittest

from provenance.data import INSTRUCTIONS, build_swapped_dataset, held_out_split


class SwappedDatasetTests(unittest.TestCase):
    def test_decoy_text_stays_within_instruction_partition(self):
        for holdout in (2, 4, 8):
            train, test = held_out_split(
                build_swapped_dataset(holdout), holdout
            )
            train_instructions = [instruction for instruction, _ in INSTRUCTIONS[:-holdout]]
            test_instructions = [instruction for instruction, _ in INSTRUCTIONS[-holdout:]]
            self.assertTrue(all(
                held not in example.text
                for example in train for held in test_instructions
            ))
            self.assertTrue(all(
                fitted not in example.text
                for example in test for fitted in train_instructions
            ))
            self.assertEqual(sum(example.trusted for example in train), len(train) // 2)
            self.assertEqual(sum(example.trusted for example in test), len(test) // 2)


if __name__ == "__main__":
    unittest.main()
