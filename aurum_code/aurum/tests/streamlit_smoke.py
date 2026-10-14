"""Real Streamlit AppTest checks, separate from the fake-Streamlit unittest process."""
from pathlib import Path
import io
import json
import sys
import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import dashboard_logic as DL


def widget(collection, suffix):
    """Return the single widget in `collection` whose key ends with `__suffix`."""
    return next(w for w in collection if w.key and w.key.endswith('__' + suffix))


def submit(app):
    """Click the model's submit button, run the app, and assert no exception was raised."""
    next(b for b in app.button if b.label == 'Run Project AURUM model').click().run()
    assert not app.exception, [e.message for e in app.exception]


app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / 'app.py'), default_timeout=120).run()
assert not app.exception
assert widget(app.number_input, 'q_min').disabled
widget(app.selectbox, 'n_max').set_value(50_000)
submit(app)
assert len(app.metric) == 8 and len(app.tabs) == 6
assert app.session_state['v2']['inp'].batch_timing is not None
assert app.session_state['v2']['inp'].yield_ord.m == 10.
original = app.session_state['v2']['res']
assert 'physical_purity_q' not in original.arrays
assert any('Probability precision:' in c.value and 'return precision:' in c.value for c in app.caption)
downloads = [b.label for b in app.get('download_button')]
assert len(downloads) == 4
assert sum('CSV schema 2' in label for label in downloads) == 2
assert any('legacy CSV schema 1' in label for label in downloads)
corrected = pd.read_csv(io.BytesIO(DL.export_csv(original)))
legacy = pd.read_csv(io.BytesIO(DL.export_csv(original, schema_version=1)))
assert corrected['physical_purity_q'].isna().all()
assert 'purity_q' not in corrected
np.testing.assert_array_equal(legacy['purity_q'], 1)
assert json.loads(DL.export_manifest(original))['export_schema_version'] == 2

# Household offsets affect adequacy, while labor and operating profit stay identical.
widget(app.radio, 'households_mode').set_value('explicit')
widget(app.number_input, 'n_hh').set_value(2)
widget(app.number_input, 'nonmining').set_value(300.)
submit(app)
state = app.session_state['v2']
assert state['inp'].n_households == 2
a = state['res'].arrays
np.testing.assert_allclose(a['B_HH'], 2*a['calendar_days_per_cycle']*500.)
for key in ('Pi', 'E', 'L_person_hours'):
    np.testing.assert_array_equal(a[key], original.arrays[key])
widget(app.number_input, 'n_hh').set_value(0)
submit(app)
np.testing.assert_array_equal(app.session_state['v2']['res'].arrays['B_HH'], 0)

# Assay metadata must not consume economic draws or alter farmgate revenue.
widget(app.radio, 'assay_mode').set_value('measured value').run()
widget(app.radio, 'households_mode').set_value('one per worker')
widget(app.number_input, 'nonmining').set_value(0.)
widget(app.radio, 'assay_unit').set_value('karat')
widget(app.number_input, 'assay_value').set_value(18.)
submit(app)
state = app.session_state['v2']
assert state['inp'].n_households is None
np.testing.assert_array_equal(state['res'].arrays['E'], original.arrays['E'])
np.testing.assert_array_equal(state['res'].arrays['physical_purity_q'], .75)

# Deterministic measurements replace inactive direct rates and retain provenance.
widget(app.radio, 'price_input').set_value('sale')
widget(app.number_input, 'sale_proceeds').set_value(2000.)
widget(app.number_input, 'sale_mass').set_value(2.)
widget(app.radio, 'fuel_input').set_value('liters × PHP/liter')
widget(app.number_input, 'fuel_liters').set_value(3.)
widget(app.number_input, 'fuel_unit_price').set_value(50.)
widget(app.text_input, 'measurement_source').set_value('synthetic AppTest measurement')
widget(app.text_input, 'measurement_date').set_value('2026-10-13')
widget(app.radio, 'p0_mode').set_value('from counts')
widget(app.number_input, 'z').set_value(2)
widget(app.number_input, 'n_obs').set_value(10)
submit(app)
state = app.session_state['v2']
a = state['res'].arrays
np.testing.assert_array_equal(a['P'], 1000.)
np.testing.assert_allclose(a['cost::fuel'], a['operating_hours']*150.)
assert state['res'].manifest['metadata']['source'] == 'synthetic AppTest measurement'
assert any('Zero-probability posterior 95% interval:' in i.value for i in app.info)
assert state['res'].manifest['numerical_diagnostics']['error_E_P95'] is not None
assert pd.read_csv(io.BytesIO(DL.export_csv(state['res'])))['physical_purity_q'].eq(.75).all()

# Reject declared overlap, incompatible price conversion and invalid assay units/ranges.
widget(app.number_input, 'fixed_cost').set_value(100.)
widget(app.text_input, 'fixed_components').set_value('Fuel')
submit(app)
assert app.error and not app.metric and 'v2' not in app.session_state
widget(app.text_input, 'fixed_components').set_value('maintenance')
submit(app)
assert not app.error
widget(app.number_input, 'assay_value').set_value(25.)
submit(app)
assert app.error and not app.metric
widget(app.number_input, 'assay_value').set_value(18.)
widget(app.radio, 'price_basis').set_value('reference').run()
submit(app)
assert app.error and not app.metric  # sale proceeds cannot become a pure-gold reference price
widget(app.radio, 'price_basis').set_value('farmgate').run()
widget(app.radio, 'assay_mode').set_value('unknown').run()
widget(app.radio, 'price_input').set_value('direct')
widget(app.radio, 'fuel_input').set_value('direct')
widget(app.radio, 'p0_mode').set_value('fixed value')
widget(app.number_input, 'fixed_cost').set_value(0.)
widget(app.text_input, 'fixed_components').set_value('')
submit(app)
assert not app.error
assert 'physical_purity_q' not in app.session_state['v2']['res'].arrays

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
print('PASS: batch default, independent/zero households and outside income, measured/unknown assay, sale/fuel conversions and metadata, posterior interval, convergence diagnostics, both export schemas, overlap/conflict clearing, all 8 reserve policies, idle time, depreciation, rental exposure, reference purity, share presets, fixed timing, invalid-yield clearing')
