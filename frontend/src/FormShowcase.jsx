import { useEffect, useState } from 'react'

/* Pre-upload showcase: a 3D account-opening form that scans itself, highlighting one
   field at a time. The stacked sheets behind the front page are there to say
   "multi-page document" at a glance, since this form is nine pages. */

const STEPS = [
  { id: 'branch', num: 1, label: 'Branch Name', caption: 'Reading the branch name' },
  { id: 'cifdate', num: 2, label: 'CIF Opening Date', caption: 'Parsing the date cells' },
  { id: 'account', num: 3, label: 'Title of Account', caption: 'Recognising the account title' },
  { id: 'cnic', num: 4, label: 'CNIC Number', caption: 'Reading the CNIC digits' },
  { id: 'gender', num: 5, label: 'Gender', caption: 'Detecting the ticked option' },
  { id: 'address', num: 6, label: 'Residential Address', caption: 'Extracting the address' },
  { id: 'property', num: 7, label: 'Property Status', caption: 'Detecting the ticked option' },
]

/* Character-grid values (dates, CNIC, IBAN) are printed one box per character on the
   real form, and reading them box-by-box is what makes them reliable — so show that. */
function Cells({ value }) {
  return (
    <span className="dcells">
      {value.split('').map((ch, i) => (
        <span key={i} className={`dcell ${ch === '-' ? 'dash' : ''}`}>{ch === '-' ? '' : ch}</span>
      ))}
    </span>
  )
}

function Box({ ticked }) {
  return (
    <span className={`dbox ${ticked ? 'tick' : ''}`}>
      {ticked && (
        <svg viewBox="0 0 24 24" fill="none" strokeWidth="3.5" strokeLinecap="round" strokeLinejoin="round">
          <path d="M4 13l5 5L20 6" />
        </svg>
      )}
    </span>
  )
}

function Field({ id, active, side = 'right', children }) {
  const step = STEPS.find((s) => s.id === id)
  return (
    <div className={`df cal-${side} ${active ? 'on' : ''}`}>
      {children}
      <span className="hl" />
      {step && (
        <span className="callout">
          <span className="c-num">{step.num}</span>
          <span className="c-label">{step.label}</span>
        </span>
      )}
    </div>
  )
}

export default function FormShowcase() {
  const [i, setI] = useState(0)
  const [inView, setInView] = useState(false)

  useEffect(() => {
    const kick = setTimeout(() => setInView(true), 120)
    const t = setInterval(() => setI((n) => (n + 1) % STEPS.length), 2400)
    return () => { clearTimeout(kick); clearInterval(t) }
  }, [])

  const cur = STEPS[i]
  const on = (id) => cur.id === id

  return (
    <div className={`showcase ${inView ? 'in' : ''}`} aria-hidden="true">
      <div className="doc-stage">
        <div className="doc">
          {/* pages behind the front sheet */}
          <span className="doc-sheet s3" />
          <span className="doc-sheet s2" />

          <div className="doc-front">
            <div className="doc-bg" />

            <div className="doc-content">
              <div className="doc-head">
                <span className="doc-logo">HBL</span>
                <span className="doc-titles">
                  <span className="doc-t1">Consumer Products Application Form</span>
                  <span className="doc-t2">Customer Information Form &middot; Page 1 of 9</span>
                </span>
              </div>

              {/* ---- For Bank Use Only ---- */}
              <div className="doc-band">
                <span className="band-cap">For Bank Use Only</span>
                <div className="drow">
                  <span className="dlab">Branch Name</span>
                  <Field id="branch" active={on('branch')} side="right">
                    <span className="dval hand">HBL PLAZA</span>
                  </Field>
                </div>
                <div className="drow">
                  <span className="dlab">CIF Opening Date</span>
                  <Field id="cifdate" active={on('cifdate')} side="right">
                    <Cells value="23062026" />
                  </Field>
                </div>
              </div>

              {/* ---- Personal Information ---- */}
              <div className="doc-sec">Personal Information</div>
              <div className="drow">
                <span className="dlab">Title of Account</span>
                <Field id="account" active={on('account')} side="right">
                  <span className="dval hand">ASHRAF CHAUDHRY</span>
                </Field>
              </div>
              <div className="drow">
                <span className="dlab">Father&rsquo;s Name</span>
                <span className="dval hand plain">Chaudry Waqar</span>
              </div>
              <div className="drow">
                <span className="dlab">CNIC / ID No.</span>
                <Field id="cnic" active={on('cnic')} side="right">
                  <Cells value="35810-0234568-5" />
                </Field>
              </div>
              <div className="drow">
                <span className="dlab">Gender</span>
                <Field id="gender" active={on('gender')} side="right">
                  <span className="dopts">
                    <span className="dopt"><Box ticked />Male</span>
                    <span className="dopt"><Box />Female</span>
                    <span className="dopt"><Box />Other</span>
                  </span>
                </Field>
              </div>

              {/* ---- Address ---- */}
              <div className="doc-sec">Residential Address</div>
              <Field id="address" active={on('address')} side="right">
                <div className="drow tight">
                  <span className="dlab">House / Street</span>
                  <span className="dval hand">H. No. 43/2, Street No. 3</span>
                </div>
                <div className="drow tight">
                  <span className="dlab">City</span>
                  <span className="dval hand">Karachi</span>
                </div>
              </Field>
              <div className="drow">
                <span className="dlab">Property Status</span>
                <Field id="property" active={on('property')} side="right">
                  <span className="dopts">
                    <span className="dopt"><Box />Owned</span>
                    <span className="dopt"><Box ticked />Rented</span>
                    <span className="dopt"><Box />Mortgaged</span>
                  </span>
                </Field>
              </div>

              <div className="doc-foot">
                <span className="dsign" />
                <span className="dlab sm">Applicant&rsquo;s Signature</span>
              </div>
            </div>

            <div className="doc-clip"><span className="scanbeam" /></div>
          </div>
        </div>
      </div>

      <div className="sc-caption">
        <span className="sc-dot" />
        <span key={i} className="sc-text">{cur.caption}</span>
      </div>
      <div className="sc-dots">
        {STEPS.map((s, n) => <span key={s.id} className={`sc-tick ${n === i ? 'on' : ''}`} />)}
      </div>
    </div>
  )
}
