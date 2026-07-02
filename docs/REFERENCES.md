# References

The methods metajudge implements are credited to their original authors here.
Every entry is carried verbatim from the decision record where the method was
adopted; the per-method rationale, and any honest caveats about applying a
method outside its original validation, live in [decisions/](decisions/). This
file is the consolidated bibliography so the credits travel with the package
rather than sitting only inside individual ADRs.

## Reliability: Krippendorff's alpha and ICC

Krippendorff, K. (2004). Content Analysis: An Introduction to Its Methodology (2nd ed.). Sage. (The ordinal alpha is computed through the `krippendorff` package; see Software below.)

Hayes, A. F., and Krippendorff, K. (2007). Answering the call for a standard reliability measure for coding data. Communication Methods and Measures, 1(1), 77-89. doi:10.1080/19312450709336664 (Basis for the percentile bootstrap CI on alpha and its small-sample caveat.)

Shrout, P. E., and Fleiss, J. L. (1979). Intraclass correlations: Uses in assessing rater reliability. Psychological Bulletin, 86(2), 420-428. doi:10.1037/0033-2909.86.2.420

McGraw, K. O., and Wong, S. P. (1996). Forming inferences about some intraclass correlation coefficients. Psychological Methods, 1(1), 30-46. doi:10.1037/1082-989X.1.1.30

Koo, T. K., and Li, M. Y. (2016). A guideline of selecting and reporting intraclass correlation coefficients for reliability research. Journal of Chiropractic Medicine, 15(2), 155-163. doi:10.1016/j.jcm.2016.02.012

## Incomplete-data reliability (outside E07 scope)

Ebel, R. L. (1951). Estimation of the reliability of ratings. Psychometrika, 16(4), 407-424. doi:10.1007/BF02288803

Nakagawa, S., and Schielzeth, H. (2010). Repeatability for Gaussian and non-Gaussian data: a practical guide for biologists. Biological Reviews, 85(4), 935-956. doi:10.1111/j.1469-185X.2010.00141.x

Brueckl, M., and Heuer, F. (2022). irrNA: Coefficients of interrater reliability, generalized for randomly incomplete datasets (R package version 0.2.3). CRAN. doi:10.32614/CRAN.package.irrNA

ten Hove, D., Jorgensen, T. D., and van der Ark, L. A. (2024). Updated guidelines on selecting an intraclass correlation coefficient for interrater reliability, with applications to incomplete observational designs. Psychological Methods, 29(5), 967-979. doi:10.1037/met0000516

ten Hove, D., Jorgensen, T. D., and van der Ark, L. A. (2025). How to estimate intraclass correlation coefficients for interrater reliability from planned incomplete data. Multivariate Behavioral Research. doi:10.1080/00273171.2025.2507745

## Differential item functioning: ordinal logistic regression

McCullagh, P. (1980). Regression models for ordinal data. Journal of the Royal Statistical Society: Series B (Methodological), 42(2), 109-142. doi:10.1111/j.2517-6161.1980.tb01109.x (The proportional-odds cumulative-logit model the DIF engine fits.)

Zumbo, B. D. (1999). A handbook on the theory and methods of differential item functioning (DIF): Logistic regression modeling as a unitary framework for binary and Likert-type (ordinal) item scores. Ottawa: Directorate of Human Resources Research and Evaluation, Department of National Defence.

Swaminathan, H., and Rogers, H. J. (1990). Detecting differential item functioning using logistic regression procedures. Journal of Educational Measurement, 27(4), 361-370. doi:10.1111/j.1745-3984.1990.tb00754.x

Crane, P. K., Gibbons, L. E., Jolley, L., and van Belle, G. (2006). Differential item functioning analysis with ordinal logistic regression techniques. Medical Care, 44(11 Suppl 3), S115-S123. doi:10.1097/01.mlr.0000245183.28384.ed

Choi, S. W., Gibbons, L. E., and Crane, P. K. (2011). lordif: An R package for detecting differential item functioning using iterative hybrid ordinal logistic regression / item response theory and Monte Carlo simulations. Journal of Statistical Software, 39(8). doi:10.18637/jss.v039.i08

Allahyari, E., Jafari, P., and Bagheri, Z. (2016). A simulation study to assess the effect of the number of response categories on the power of ordinal logistic regression for differential item functioning analysis in rating scales. Computational and Mathematical Methods in Medicine, 2016, 5080826. doi:10.1155/2016/5080826

### Effect size and classification

Nagelkerke, N. J. D. (1991). A note on a general definition of the coefficient of determination. Biometrika, 78(3), 691-692. doi:10.1093/biomet/78.3.691 (The pseudo-R-squared whose change between nested models is the DIF effect size.)

Jodoin, M. G., and Gierl, M. J. (2001). Evaluating Type I error and power rates using an effect size measure with the logistic regression procedure for DIF detection. Applied Measurement in Education, 14(4), 329-349. doi:10.1207/s15324818ame1404_2

Gomez-Benito, J., Hidalgo, M. D., and Padilla, J. L. (2009). Efficacy of effect size measures in logistic regression: An application for detecting DIF. Methodology, 5(1), 18-25. doi:10.1027/1614-2241.5.1.18

### Matching-criterion contamination and purification

Holland, P. W., and Thayer, D. T. (1986). Differential item functioning and the Mantel-Haenszel procedure. ETS Research Report Series, 1986(2). doi:10.1002/j.2330-8516.1986.tb00186.x

Dorans, N. J., and Holland, P. W. (1992). DIF detection and description: Mantel-Haenszel and standardization. ETS Research Report Series, 1992(1). doi:10.1002/j.2333-8504.1992.tb01440.x

Clauser, B. E., Mazor, K. M., and Hambleton, R. K. (1993). The effects of purification of the matching criterion on the identification of DIF using the Mantel-Haenszel procedure. Applied Measurement in Education, 6(4), 269-279. doi:10.1207/s15324818ame0604_2

French, B. F., and Maller, S. J. (2007). Iterative purification and effect size use with logistic regression for differential item functioning detection. Educational and Psychological Measurement, 67(3), 373-393. doi:10.1177/0013164406294781

Magis, D., Beland, S., Tuerlinckx, F., and De Boeck, P. (2010). A general framework and an R package for the detection of dichotomous differential item functioning. Behavior Research Methods, 42(3), 847-862. doi:10.3758/brm.42.3.847

### Proportional-odds assumption diagnostic

Brant, R. (1990). Assessing proportionality in the proportional odds model for ordinal logistic regression. Biometrics, 46(4), 1171-1178. doi:10.2307/2532457 (The Wald-type test `dif._brant_test` implements.)

Harrell, F. E. (2015). Regression Modeling Strategies (2nd ed.). Springer. doi:10.1007/978-3-319-19425-7 (Source for the Brant-test oversensitivity caveat at large N surfaced in the report card.)

## Clustered-error inference (the cluster bootstrap)

Liang, K.-Y., and Zeger, S. L. (1986). Longitudinal data analysis using generalized linear models. Biometrika, 73(1), 13-22.

den Noortgate, W. V., and De Boeck, P. (2005). Assessing and explaining differential item functioning using logistic mixed models. Journal of Educational and Behavioral Statistics, 30(4), 443-464.

French, B. F., and Finch, W. H. (2010). Hierarchical logistic regression: Accounting for multilevel data in DIF detection. Journal of Educational Measurement, 47(3), 299-317.

Musca, S. C., et al. (2011). Data with hierarchical structure: impact of intraclass correlation and sample size on Type-I error. Frontiers in Psychology, 2:74.

Cameron, A. C., and Miller, D. L. (2015). A practitioner's guide to cluster-robust inference. Journal of Human Resources, 50(2), 317-372.

## Rater-mediated measurement and LLM-as-judge evaluation

Cronbach, L. J. (1951). Coefficient alpha and the internal structure of tests. Psychometrika, 16(3), 297-334. doi:10.1007/BF02310555

Cortina, J. M. (1993). What is coefficient alpha? An examination of theory and applications. Journal of Applied Psychology, 78(1), 98-104. doi:10.1037/0021-9010.78.1.98

Brennan, R. L. (2001). Generalizability theory. Springer. doi:10.1007/978-1-4757-3456-0_6

Lim, G. S. (2011). The development and maintenance of rating quality in performance writing assessment: A longitudinal study of new and experienced raters. Language Testing, 28(4), 543-560. doi:10.1177/0265532211406422

Hallgren, K. A. (2012). Computing inter-rater reliability for observational data: An overview and tutorial. Tutorials in Quantitative Methods for Psychology, 8(1), 23-34. doi:10.20982/tqmp.08.1.p023

McNeish, D. (2018). Thanks coefficient alpha, we'll take it from here. Psychological Methods, 23(3), 412-433. doi:10.1037/met0000144

Wang, J., and Luo, K. (2019). Evaluating rater judgments on ETIC Advanced writing tasks: An application of generalizability theory and many-facet Rasch. Studies in Language Assessment. doi:10.58379/vmak1620

Li, W. (2022). Scoring rubric reliability and internal validity in rater-mediated EFL writing assessment: Insights from many-facet Rasch measurement. Reading and Writing, 35(10), 2409-2431. doi:10.1007/s11145-022-10279-1

Zheng, L., Chiang, W.-L., Sheng, Y., et al. (2023). Judging LLM-as-a-judge with MT-Bench and Chatbot Arena. arXiv:2306.05685.

Norman, J. D., Rivera, M. U., and Hughes, D. A. (2026). Reliability without validity: A systematic, large-scale evaluation of LLM-as-a-judge models across agreement, consistency, and bias. arXiv:2606.19544.

Wang, P., Li, L., Chen, L., Cai, Z., Zhu, D., Lin, B., Cao, Y., Liu, Q., Liu, T., and Sui, Z. (2023). Large language models are not fair evaluators. arXiv:2305.17926. Published at ACL 2024 (Volume 1: Long Papers). doi:10.18653/v1/2024.acl-long.511 (Position bias in LLM-as-judge scoring; motivates auditing the judge instrument, not just the scored model.)

Bavaresco, A., Bernardi, R., Bertolazzi, L., Elliott, D., Fernandez, R., Gatt, A., Ghaleb, E., Giulianelli, M., Hanna, M., Koller, A., Martins, A. F. T., Mondorf, P., Neplenbroek, V., Pezzelle, S., Plank, B., Schlangen, D., Suglia, A., Surikuchi, A. K., Takmaz, E., and Testoni, A. (2024). LLMs instead of human judges? A large scale empirical study across 20 NLP evaluation tasks. arXiv:2406.18403. Accepted to ACL 2025 main conference.

Bachmann, D., van der Wal, O., Chvojka, E., Zuidema, W. H., van Maanen, L., and Schulz, K. (2024). fl-IRT-ing with psychometrics to improve NLP bias measurement. Minds and Machines, 34(4), 37. doi:10.1007/s11023-024-09695-9

Xu, Z., Kandanaarachchi, S., Ong, C. S., and Ntoutsi, E. (2025). Fairness evaluation with item response theory. Proceedings of the ACM Web Conference 2025 (WWW '25). arXiv:2411.02414. doi:10.1145/3696410.3714883

Choi, J., Park, S., Cho, C., Park, H., and Kim, B. (2026). Diagnosing the reliability of LLM-as-a-judge via item response theory. arXiv:2602.00521. Accepted ICML 2026. (IRT/GRM-based judge-reliability diagnostic; explicitly defers DIF-style stratum comparisons to future work, per the SC2 re-check logged in `docs/decisions/`.)

## Demo corpus

Fabbri, A. R., Kryscinski, W., McCann, B., Xiong, C., Socher, R., and Radev, D. (2020). SummEval: Re-evaluating Summarization Evaluation. arXiv:2007.12626. (Redistributed under MIT; see [`src/metajudge/data/SOURCE.md`](../src/metajudge/data/SOURCE.md).)

## Software and reference implementations

The runtime alpha computation wraps the `krippendorff` Python package. The
statistics are pinned in the test suite against external reference
implementations: R `MASS::polr` for the proportional-odds DIF fit, and the
`pingouin` and `statsmodels` Python packages as ICC and logistic-regression
oracles. `pingouin` and `statsmodels` are test-only oracles and are never
imported at runtime.

AutoRubric: a Python library for rubric-based LLM-as-judge grading with
agreement metrics and bootstrap confidence intervals (https://autorubric.org,
https://github.com/delip/autorubric). Adjacent, shipped practitioner coverage
of the commodity reliability layer; cited in SPEC.md as prior art this
library does not compete with.
