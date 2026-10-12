"""Real Streamlit AppTest checks, separate from the fake-Streamlit unittest process."""
from pathlib import Path
import numpy as np
from streamlit.testing.v1 import AppTest


def widget(collection, suffix):
    return next(w for w in collection if w.key and w.key.endswith('__' + suffix))


def submit(app):
    next(b for b in app.button if b.label == 'Run Project AURUM model').click().run()
    assert not app.exception, [e.message for e in app.exception]


app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / 'app.py'), default_timeout=120).run()
assert not app.exception
assert widget(app.number_input, 'q_min').disabled
submit(app)
assert len(app.metric) == 8 and len(app.tabs) == 6
assert app.session_state['v2']['inp'].batch_timing is not None
assert app.session_state['v2']['inp'].yield_ord.m == 10.
widget(app.number_input, 'days_1').set_value(5)
submit(app)
assert app.error and not app.metric
assert 'v2' not in app.session_state
widget(app.number_input, 'days_1').set_value(6)
widget(app.number_input, 'idle_min').set_value(2.5)
widget(app.number_input, 'idle_mode').set_value(2.5)
widget(app.number_input, 'idle_max').set_value(2.5)
widget(app.radio, 'reserve_mode').set_value('custom')
widget(app.number_input, 'reserve_custom').set_value(500.)
submit(app)
assert app.error and not app.metric
for base in ('gross', 'net'):
    for location in ('before_sharing', 'after_sharing'):
        for priority in ('household_first', 'reserve_first'):
            widget(app.radio, 'tribute_base').set_value(base)
            widget(app.radio, 'reserve_location').set_value(location)
            widget(app.radio, 'reserve_priority').set_value(priority)
            submit(app)
            assert not app.error and len(app.metric) == 8
            state = app.session_state['v2']
            assert state['inp'].reserve_location == location
            assert state['inp'].reserve_priority == priority
            np.testing.assert_allclose(state['res'].arrays['idle_days'], 2.5)
widget(app.radio, 'reserve_mode').set_value('match_depreciation')
widget(app.radio, 'eq_mode').set_value('purchased')
widget(app.number_input, 'eq_acq').set_value(10000.)
submit(app)
assert not app.error
assert app.session_state['v2']['res'].summary['reserve_planned_mean'] > 0
widget(app.radio, 'price_basis').set_value('reference').run()
assert not widget(app.number_input, 'q_min').disabled
widget(app.radio, 'share_preset').set_value('60/40').run()
assert widget(app.number_input, 's_m').disabled
submit(app)
assert not app.error
assert app.session_state['v2']['inp'].s_m == .6
assert app.session_state['v2']['res'].manifest['valuation_mass_label'] == 'fine gold mass'
widget(app.radio, 'eq_mode').set_value('rented')
widget(app.number_input, 'eq_rent').set_value(100.)
submit(app)
a = app.session_state['v2']['res'].arrays
np.testing.assert_array_equal(a['cost::equipment:main_equipment'], a['total_shifts'] * 100)
widget(app.radio, 'timing_mode').set_value('fixed').run()
widget(app.radio, 'explosives_basis').set_value('per_batch')
widget(app.number_input, 'calendar_days').set_value(2.5)
submit(app)
assert not app.error
assert app.session_state['v2']['res'].summary['calendar_days_per_cycle'] == 2.5
widget(app.number_input, 'ya').set_value(2.)
widget(app.number_input, 'yb').set_value(1.)
submit(app)
assert app.error and not app.metric
print('PASS: batch default, invalid-count clearing, all 8 reserve policies, idle time, depreciation, rental exposure, reference purity, share presets, fixed timing, invalid-yield clearing')
