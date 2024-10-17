/**
 * name: @9am/before-after@1.2.0
 * desc: A small Web Component shows a before-after comparison.
 * author: 9am <tech.9am@gmail.com> [https://9am.github.io/]
 * homepage: https://9am.github.io/before-after/
 * license: MIT
 */
var c = (t, a, e) => {
  if (!a.has(t))
    throw TypeError("Cannot " + e);
};
var r = (t, a, e) => (c(t, a, "read from private field"), e ? e.call(t) : a.get(t)), l = (t, a, e) => {
  if (a.has(t))
    throw TypeError("Cannot add the same private member more than once");
  a instanceof WeakSet ? a.add(t) : a.set(t, e);
}, h = (t, a, e, o) => (c(t, a, "write to private field"), o ? o.call(t, e) : a.set(t, e), e);
var d = (t, a, e) => (c(t, a, "access private method"), e);
const b = `*,*:before,*:after{box-sizing:border-box}:host{--position: to right;--repeat: 1;--thumb-size: 1px;--thumb-color: white;--val: 0;--p: calc(var(--val) / var(--repeat) * 1%);display:grid;position:relative;overflow:visible;--thumb-size-hf: calc(var(--thumb-size) / 2);--thumb-gradient-val: var(--position), #0000 0 calc(var(--p) - var(--thumb-size-hf)), var(--thumb-color) calc(var(--p) - var(--thumb-size-hf)) calc(var(--p) + var(--thumb-size-hf)), #0000 0 calc(100% / var(--repeat));--thumb-opacity: calc(var(--val) * (100 - var(--val)));--mask-before-val: var(--position), #0000 0, var(--p), #000 0 calc(100% / var(--repeat));--mask-after-val: var(--position), #000 0, var(--p), #0000 0 calc(100% / var(--repeat))}:host:after{content:"thumb";font-size:0;position:absolute;top:0;left:0;width:100%;height:100%;filter:drop-shadow(0px 0px 1px black);background:var(--thumb-gradient);opacity:var(--thumb-opacity);pointer-events:none}:host(:hover):after{filter:drop-shadow(0px 0px 2px black)}::slotted([slot=before]){mask:var(--mask-before)}::slotted([slot=after]){mask:var(--mask-after)}#range,::slotted([slot=before]),::slotted([slot=after]){grid-area:1 / 1}#range{z-index:1;opacity:0;-moz-appearance:none;appearance:none;-webkit-appearance:none;background:transparent;cursor:ew-resize}#range:focus{outline:none}#range::-webkit-slider-thumb{-moz-appearance:none;appearance:none;-webkit-appearance:none;width:0px;height:0px}:host([orient=vertical]){--position: to bottom}:host([orient=vertical]) #range[orient=vertical]{-webkit-appearance:slider-vertical;-moz-appearance:slider-vertical;appearance:slider-vertical;cursor:ns-resize;transform:rotate(.5turn)}:host(:not([varient*=repeating])){--repeat: 1 !important}:host([varient*=repeating]){--repeat: 6}:host([varient*=radial]){--position: circle at center}:host([varient*=conic]){--position: at center;--thumb-size: .2deg;--p: calc(var(--val) / var(--repeat) * .01turn)}:host{--thumb-gradient: linear-gradient(var(--thumb-gradient-val));--mask-before: linear-gradient(var(--mask-before-val));--mask-after: linear-gradient(var(--mask-after-val))}:host([varient=radial]){--thumb-gradient: radial-gradient(var(--thumb-gradient-val));--mask-before: radial-gradient(var(--mask-before-val));--mask-after: radial-gradient(var(--mask-after-val))}:host([varient=conic]){--thumb-gradient: conic-gradient(var(--thumb-gradient-val));--mask-before: conic-gradient(var(--mask-before-val));--mask-after: conic-gradient(var(--mask-after-val))}:host([varient=repeating-linear]){--thumb-gradient: repeating-linear-gradient(var(--thumb-gradient-val));--mask-before: repeating-linear-gradient(var(--mask-before-val));--mask-after: repeating-linear-gradient(var(--mask-after-val))}:host([varient=repeating-radial]){--thumb-gradient: repeating-radial-gradient(var(--thumb-gradient-val));--mask-before: repeating-radial-gradient(var(--mask-before-val));--mask-after: repeating-radial-gradient(var(--mask-after-val))}:host([varient=repeating-conic]){--thumb-gradient: repeating-conic-gradient(var(--thumb-gradient-val));--mask-before: repeating-conic-gradient(var(--mask-before-val));--mask-after: repeating-conic-gradient(var(--mask-after-val))}
`, m = new CSSStyleSheet();
m.replaceSync(b);
var i, v, u, s;
class g extends HTMLElement {
  constructor() {
    super();
    l(this, v);
    l(this, i, null);
    l(this, s, () => {
      var e;
      this.style.setProperty("--val", `${((e = r(this, i)) == null ? void 0 : e.value) ?? 0}`);
    });
    this.attachShadow({ mode: "open" }), this.shadowRoot.adoptedStyleSheets = [m], this.shadowRoot.append(
      d(this, v, u).call(this, {
        value: this.value,
        orient: this.orient
      }).content.cloneNode(!0)
    ), h(this, i, this.shadowRoot.querySelector("#range")), r(this, s).call(this);
  }
  static get observedAttributes() {
    return ["orient", "value"];
  }
  attributeChangedCallback(e, o, n) {
    var p;
    if (o !== n)
      switch (e) {
        case "orient": {
          (p = r(this, i)) == null || p.setAttribute("orient", n);
          break;
        }
        case "value": {
          this.value = n, r(this, i).value = n, r(this, i).dispatchEvent(new InputEvent("input"));
          break;
        }
      }
  }
  connectedCallback() {
    var e;
    (e = r(this, i)) == null || e.addEventListener("input", r(this, s));
  }
  disconnectedCallback() {
    var e;
    (e = r(this, i)) == null || e.removeEventListener("input", r(this, s));
  }
  get orient() {
    return this.getAttribute("orient") ?? "horizontal";
  }
  set orient(e) {
    this.setAttribute("orient", e);
  }
  get value() {
    return this.getAttribute("value") ?? 50;
  }
  set value(e) {
    this.setAttribute("value", e);
  }
}
i = new WeakMap(), v = new WeakSet(), u = function({ value: e, orient: o }) {
  const n = document.createElement("template");
  return n.innerHTML = `
    <input type="range" id="range" part="range" orient="${o}" value=${e} min="0" max="100" step="any" />
    <slot name="before"></slot>
    <slot name="after"></slot>
`, n;
}, s = new WeakMap();
window && !window.customElements.get("before-after") && window.customElements.define("before-after", g);
const k = null;
export {
  g as BeforeAfter,
  k as default
};
