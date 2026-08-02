The  dataset.csv header
AnonymizationID,MRN,Study UID,Series UID (Ax MAGiC),dicom_path,synthentic_path,T1W Synthetic,T2W Synthetic,FLAIR Synthetic,SYMAPS,PS Synthetic,match_status,n_matches_for_MRN,n_matches_for_StudyUID,n_matches_for_MRN_StudyUID

has been updated with the path to the T1 T2 PD maps. as

update the workflow according to the file structure below

$ ls $synthentic_path/$SYMAPS
SYMAPS_01_PD.dcm  SYMAPS_03_T1.dcm  SYMAPS_05_T2.dcm  SYMAPS_08_PD.dcm  SYMAPS_10_T1.dcm  SYMAPS_12_T2.dcm  SYMAPS_15_PD.dcm  SYMAPS_17_T1.dcm  SYMAPS_19_T2.dcm  SYMAPS_22_PD.dcm  SYMAPS_24_T1.dcm  SYMAPS_26_T2.dcm  SYMAPS_29_PD.dcm
SYMAPS_01_T1.dcm  SYMAPS_03_T2.dcm  SYMAPS_06_PD.dcm  SYMAPS_08_T1.dcm  SYMAPS_10_T2.dcm  SYMAPS_13_PD.dcm  SYMAPS_15_T1.dcm  SYMAPS_17_T2.dcm  SYMAPS_20_PD.dcm  SYMAPS_22_T1.dcm  SYMAPS_24_T2.dcm  SYMAPS_27_PD.dcm  SYMAPS_29_T1.dcm
SYMAPS_01_T2.dcm  SYMAPS_04_PD.dcm  SYMAPS_06_T1.dcm  SYMAPS_08_T2.dcm  SYMAPS_11_PD.dcm  SYMAPS_13_T1.dcm  SYMAPS_15_T2.dcm  SYMAPS_18_PD.dcm  SYMAPS_20_T1.dcm  SYMAPS_22_T2.dcm  SYMAPS_25_PD.dcm  SYMAPS_27_T1.dcm  SYMAPS_29_T2.dcm
SYMAPS_02_PD.dcm  SYMAPS_04_T1.dcm  SYMAPS_06_T2.dcm  SYMAPS_09_PD.dcm  SYMAPS_11_T1.dcm  SYMAPS_13_T2.dcm  SYMAPS_16_PD.dcm  SYMAPS_18_T1.dcm  SYMAPS_20_T2.dcm  SYMAPS_23_PD.dcm  SYMAPS_25_T1.dcm  SYMAPS_27_T2.dcm  SYMAPS_30_PD.dcm
SYMAPS_02_T1.dcm  SYMAPS_04_T2.dcm  SYMAPS_07_PD.dcm  SYMAPS_09_T1.dcm  SYMAPS_11_T2.dcm  SYMAPS_14_PD.dcm  SYMAPS_16_T1.dcm  SYMAPS_18_T2.dcm  SYMAPS_21_PD.dcm  SYMAPS_23_T1.dcm  SYMAPS_25_T2.dcm  SYMAPS_28_PD.dcm  SYMAPS_30_T1.dcm
SYMAPS_02_T2.dcm  SYMAPS_05_PD.dcm  SYMAPS_07_T1.dcm  SYMAPS_09_T2.dcm  SYMAPS_12_PD.dcm  SYMAPS_14_T1.dcm  SYMAPS_16_T2.dcm  SYMAPS_19_PD.dcm  SYMAPS_21_T1.dcm  SYMAPS_23_T2.dcm  SYMAPS_26_PD.dcm  SYMAPS_28_T1.dcm  SYMAPS_30_T2.dcm
SYMAPS_03_PD.dcm  SYMAPS_05_T1.dcm  SYMAPS_07_T2.dcm  SYMAPS_10_PD.dcm  SYMAPS_12_T1.dcm  SYMAPS_14_T2.dcm  SYMAPS_17_PD.dcm  SYMAPS_19_T1.dcm  SYMAPS_21_T2.dcm  SYMAPS_24_PD.dcm  SYMAPS_26_T1.dcm  SYMAPS_28_T2.dcm
