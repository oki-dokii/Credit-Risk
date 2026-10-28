/* Set bridge_dir to the directory containing design.csv. */
%let bridge_dir=/absolute/path/to/reconciliation;
data design;
  infile "&bridge_dir./design.csv" dsd dlm=',' firstobs=2 lrecl=32767 truncover;
  length split $24;
  input row_id :best32. split :$24. target :best32. w001 :best32. w002 :best32. w003 :best32. w004 :best32. w005 :best32. w006 :best32.;
run;
ods output ParameterEstimates=parameters;
proc logistic data=design(where=(split='train'));
  model target(event='1') = w001 w002 w003 w004 w005 w006 / maxiter=1000 gconv=1e-10;
  score data=design out=scored;
run;
ods output close;
data sas_predictions;
  set scored;
  sas_pd=P_1;
  keep row_id split target w001 w002 w003 w004 w005 w006 sas_pd;
  format _numeric_ best32.;
run;
data sas_coefficients;
  set parameters;
  length term $32;
  term=Variable;
  keep term estimate;
  format estimate best32.;
run;
proc export data=sas_predictions outfile="&bridge_dir./sas_predictions.csv" dbms=csv replace; run;
proc export data=sas_coefficients outfile="&bridge_dir./sas_coefficients.csv" dbms=csv replace; run;
/* Check the SAS log for convergence, separation, and discarded observations.
   Do not report a pass when SAS has issued fitting warnings. */
