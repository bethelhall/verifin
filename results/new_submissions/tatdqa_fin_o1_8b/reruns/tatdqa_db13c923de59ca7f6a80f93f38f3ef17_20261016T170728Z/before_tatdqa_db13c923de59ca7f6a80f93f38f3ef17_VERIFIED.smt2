(set-logic QF_NRA)
(set-option :produce-unsat-cores true)
(set-option :produce-models true)

(declare-const computed_percent_change Real)
(declare-const segment_operating_income_2017 Real)
(declare-const segment_operating_income_2018 Real)

(assert (! (= segment_operating_income_2017 147.69999999999999) :named evidence_segment_operating_income_2017))
(assert (! (= segment_operating_income_2018 234.5) :named evidence_segment_operating_income_2018))

(assert (! (= computed_percent_change (* (/ (- segment_operating_income_2018 segment_operating_income_2017) segment_operating_income_2017) 100)) :named formula_percent_change))
(assert (! (or (> segment_operating_income_2017 0) (< segment_operating_income_2017 0)) :named domain_percent_change_0))

(assert (! (<= (- computed_percent_change 58.770000000000003) 0.058770000000000003) :named claim_upper))
(assert (! (<= (- 58.770000000000003 computed_percent_change) 0.058770000000000003) :named claim_lower))

(check-sat)
(get-unsat-core)
(get-model)