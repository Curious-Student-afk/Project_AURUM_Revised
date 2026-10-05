"""
A minimal stand-in for the `streamlit` module, used ONLY to smoke-test app.py where Streamlit is not installed.

It records every call, returns each widget's default (or an override), and enforces the same argument rules that
commonly raise errors in real Streamlit: mixed int/float numeric arguments, defaults outside min/max, duplicate widget
keys, and selectbox/radio defaults that are not among the options.

Limits: it cannot check layout, rendering or version-specific API changes. Run `streamlit run app.py` for that.
"""
import importlib
import sys
import types


class StopApp(Exception):
    pass


class _Ctx:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class _State(dict):
    def __getattr__(self, k):
        try:
            return self[k]
        except KeyError:
            raise AttributeError(k)

    def __setattr__(self, k, v):
        self[k] = v


def _check_numeric(label, value, min_value, max_value, step, override_present):
    nums = [x for x in (value, min_value, max_value, step) if x is not None]
    kinds = {float if isinstance(x, float) else int for x in nums if not isinstance(x, bool)}
    if len(kinds) > 1:
        raise TypeError(f"Mixed numeric types in widget '{label}': value={value!r} min={min_value!r} "
                        f"max={max_value!r} step={step!r}")
    if not override_present:
        if min_value is not None and value < min_value:
            raise ValueError(f"Default {value!r} below min {min_value!r} in '{label}'")
        if max_value is not None and value > max_value:
            raise ValueError(f"Default {value!r} above max {max_value!r} in '{label}'")


class Fake:
    def __init__(self, overrides=None, submit=True, press=(), session_state=None):
        self.log = []
        self.overrides = overrides or {}
        self.submit = submit
        self.press = set(press)
        self.session_state = session_state if session_state is not None else _State()
        self._keys = set()
        self.sidebar = _Sidebar(self)

    # ---- helpers ----
    def _rec(self, kind, *args, **kw):
        self.log.append((kind, args, kw))

    def _override(self, key):
        if key is None:
            return False, None
        name = key.split("__")[-1]
        for k in (key, name):
            if k in self.overrides:
                return True, self.overrides[k]
        return False, None

    def _register(self, key):
        if key is not None:
            if key in self._keys:
                raise RuntimeError(f"DuplicateWidgetID: {key}")
            self._keys.add(key)

    def texts(self):
        out = []
        for kind, args, kw in self.log:
            if kind in ("markdown", "caption", "subheader", "header", "title", "info", "warning", "error", "success",
                        "write", "code", "metric", "download_button", "button", "expander"):
                out.append(" ".join(str(a) for a in args if isinstance(a, (str, int, float))))
        return out

    def calls(self, kind):
        return [(a, k) for kd, a, k in self.log if kd == kind]

    # ---- layout ----
    def set_page_config(self, **kw): self._rec("set_page_config", **kw)
    def spinner(self, *a, **k): return _Ctx()
    def expander(self, label, expanded=False): self._rec("expander", label); return _Ctx()
    def columns(self, spec):
        n = spec if isinstance(spec, int) else len(spec)
        return [_Ctx() for _ in range(n)]
    def tabs(self, labels): self._rec("tabs", *labels); return [_Ctx() for _ in labels]
    def form(self, key=None, **k): self._register("form::" + str(key)); return _Ctx()
    def stop(self): raise StopApp()

    # ---- display ----
    def _disp(kind):
        def f(self, *a, **k): self._rec(kind, *a, **k)
        return f
    title = _disp("title"); markdown = _disp("markdown"); caption = _disp("caption"); subheader = _disp("subheader")
    header = _disp("header"); info = _disp("info"); warning = _disp("warning"); error = _disp("error")
    success = _disp("success"); write = _disp("write"); code = _disp("code"); metric = _disp("metric")
    dataframe = _disp("dataframe"); pyplot = _disp("pyplot"); download_button = _disp("download_button")

    # ---- widgets ----
    def number_input(self, label, min_value=None, max_value=None, value=0, step=None, key=None, help=None, format=None):
        self._register(key)
        ok, ov = self._override(key)
        _check_numeric(label, value, min_value, max_value, step, ok)
        if format is not None and not isinstance(format, str):
            raise TypeError("format must be str")
        self._rec("number_input", label, key=key)
        return ov if ok else value

    def slider(self, label, min_value=None, max_value=None, value=None, step=None, key=None, help=None):
        self._register(key)
        ok, ov = self._override(key)
        _check_numeric(label, value, min_value, max_value, step, ok)
        self._rec("slider", label, key=key)
        return ov if ok else value

    def selectbox(self, label, options, index=0, key=None, help=None):
        self._register(key)
        options = list(options)
        if not 0 <= index < len(options):
            raise ValueError(f"selectbox index out of range in '{label}'")
        ok, ov = self._override(key)
        self._rec("selectbox", label, key=key)
        return ov if ok else options[index]

    def radio(self, label, options, index=0, key=None, help=None):
        return self.selectbox(label, options, index=index, key=key, help=help)

    def text_input(self, label, value="", key=None, help=None):
        self._register(key)
        ok, ov = self._override(key)
        self._rec("text_input", label, key=key)
        return ov if ok else value

    def form_submit_button(self, label, *a, **k):
        self._rec("form_submit_button", label)
        return self.submit(label) if callable(self.submit) else bool(self.submit)

    def button(self, label, *a, **k):
        self._rec("button", label)
        return label in self.press


class _Sidebar:
    def __init__(self, parent):
        self._p = parent

    def __getattr__(self, name):
        return getattr(self._p, name)


def run_app(overrides=None, submit=True, press=(), session_state=None):
    """Import a fresh app.py bound to a Fake streamlit and run main(). Returns (fake, stopped)."""
    fake = Fake(overrides, submit, press, session_state)
    mod = types.ModuleType("streamlit")
    for name in dir(fake):
        if not name.startswith("_"):
            setattr(mod, name, getattr(fake, name))
    mod.session_state = fake.session_state
    sys.modules["streamlit"] = mod
    sys.modules.pop("app", None)
    app = importlib.import_module("app")
    stopped = False
    try:
        app.main()
    except StopApp:
        stopped = True
    return fake, stopped
