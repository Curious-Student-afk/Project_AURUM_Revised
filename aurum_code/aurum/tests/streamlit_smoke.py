"""Real Streamlit runtime smoke; run separately from the fake-Streamlit unittest suite."""
from pathlib import Path
from streamlit.testing.v1 import AppTest


def widget(collection, suffix):
    return next(w for w in collection if w.key and w.key.endswith('__' + suffix))


def submit(app):
    next(b for b in app.button if b.label == 'Run Project AURUM model').click().run()
    assert not app.exception, [e.message for e in app.exception]


app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / 'app.py'), default_timeout=120).run()
assert not app.exception
submit(app)
assert len(app.metric) == 8
assert len(app.tabs) == 6
widget(app.number_input, 'calendar_days').set_value(2.5)
widget(app.radio, 'reserve_mode').set_value('custom')
widget(app.number_input, 'reserve_custom').set_value(500.)
submit(app)
assert app.error and not app.metric  # enabled reserve requires explicit policies
for base in ('gross', 'net'):
    for location in ('before_sharing', 'after_sharing'):
        for priority in ('household_first', 'reserve_first'):
            widget(app.radio, 'tribute_base').set_value(base)
            widget(app.radio, 'reserve_location').set_value(location)
            widget(app.radio, 'reserve_priority').set_value(priority)
            submit(app)
            assert not app.error
            assert len(app.metric) == 8
            state = app.session_state['v2']
            assert state['inp'].calendar_days_per_cycle == 2.5
            assert state['inp'].reserve_location == location
            assert state['inp'].reserve_priority == priority
            assert state['res'].manifest['inputs']['reserve_mode'] == 'custom'
widget(app.radio, 'reserve_mode').set_value('match_depreciation')
widget(app.radio, 'eq_mode').set_value('purchased')
widget(app.number_input, 'eq_acq').set_value(10000.)
submit(app)
assert not app.error
assert app.session_state['v2']['res'].summary['reserve_planned_mean'] > 0
widget(app.number_input, 'ya').set_value(2.)
widget(app.number_input, 'yb').set_value(1.)
submit(app)
assert app.error and not app.metric
print('PASS: default, explicit-policy errors, all 8 reserve combinations, calendar duration, depreciation reserve, invalid-yield clearing')
