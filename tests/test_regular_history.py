import unittest
from scripts.run_regular_history import gap_profile, profiles_at_origin


class RegularHistoryTests(unittest.TestCase):
    cfg={'recent_gap_count':5,'minimum_past_gaps':3,'maximum_gap_cv':.5}

    def test_user_examples_and_insufficient_intervals(self):
        self.assertTrue(gap_profile([14,21,28,14],self.cfg)['regular'])
        self.assertFalse(gap_profile([2,60,2,60],self.cfg)['regular'])
        self.assertEqual(gap_profile([30],self.cfg)['status'],'insufficient')
        self.assertEqual(gap_profile([30,30],self.cfg)['status'],'insufficient')
        self.assertTrue(gap_profile([30,30,30],self.cfg)['regular'])

    def test_future_gap_does_not_select_past_origin_and_hold_resets(self):
        def e(i,day,state='completed',product='p'):
            return dict(event_id=i,ordered_on=day,event_state=state,product_id=product,customer_id='c')
        past=[e('a','2026-01-01'),e('b','2026-01-15'),e('c','2026-01-29'),e('d','2026-02-12')]
        before=profiles_at_origin(past,self.cfg)
        after=profiles_at_origin(past+[e('x','2026-02-13',product='other'),e('f','2026-04-13'),e('h','2026-04-14','held'),e('i','2026-04-15')],self.cfg)
        self.assertTrue(before['d']['regular'])
        self.assertEqual(before['d'],after['d'])
        self.assertEqual(after['i']['status'],'insufficient')


if __name__=='__main__':unittest.main()
