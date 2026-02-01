# Model Comparison Report: XGBoost Text-Only vs OSINT-Enhanced

## 📄 Reference Paper

**An et al. (2025)**  
*Multilingual Email Phishing Attacks Detection using OSINT and Machine Learning*  
🔗 https://arxiv.org/html/2501.08723v1

---

## 📊 Performance Comparison

### Results Table

| Model             |   Accuracy (%) |   F1 Score (%) |   Recall (%) |   Precision (%) |   Features |   Paper Target Acc (%) |
|:------------------|---------------:|---------------:|-------------:|----------------:|-----------:|-----------------------:|
| XGBoost Text-Only |        90.4282 |        90.2875 |      88.0923 |         92.595  |        159 |                  95.39 |
| XGBoost + OSINT   |        91.5302 |        91.3254 |      88.2793 |         94.5892 |        173 |                  96.71 |

### 📈 OSINT Enhancement Impact

- **Accuracy**: +1.10%
- **Precision**: +1.99%
- **Recall**: +0.19%
- **F1 Score**: +1.04%

### 📚 Comparison with Paper

| Metric | Our Result | Paper Result |
|--------|------------|-------------|
| XGBoost Text-Only Accuracy | 90.43% | 95.39% |
| XGBoost + OSINT Accuracy | 91.53% | 96.71% |
| Improvement | +1.10% | +1.32% |

---

## 🔍 Confusion Matrix Analysis

### Text-Only

```
              Predicted
              Safe  Phishing
Actual Safe     1459     113
       Phishing  191    1413
```

- True Negatives (TN): 1459
- False Positives (FP): 113
- False Negatives (FN): 191
- True Positives (TP): 1413

### OSINT-Enhanced

```
              Predicted
              Safe  Phishing
Actual Safe     1491      81
       Phishing  188    1416
```

- True Negatives (TN): 1491
- False Positives (FP): 81
- False Negatives (FN): 188
- True Positives (TP): 1416

---

## 🎯 Feature Analysis

### Text-Only Features (159 features)

- `n_urls`
- `n_domains`
- `n_ips`
- `has_https`
- `has_http`
- `n_suspicious_keywords`
- `email_length`
- `n_attachments`
- `count_suspicious_tld`
- `tfidf_000`
- `tfidf_0000`
- `tfidf_0100`
- `tfidf_0500`
- `tfidf_06`
- `tfidf_06 aug`
- `tfidf_07`
- `tfidf_07 aug`
- `tfidf_08`
- `tfidf_0800`
- `tfidf_10`
- `tfidf_100`
- `tfidf_20`
- `tfidf_2002`
- `tfidf_2008`
- `tfidf_24`
- `tfidf_2577781`
- `tfidf_2577781 added`
- `tfidf________________________________________________`
- `tfidf_account`
- `tfidf_added`
- `tfidf_added submissionid`
- `tfidf_address`
- `tfidf_aug`
- `tfidf_aug 2008`
- `tfidf_best`
- `tfidf_business`
- `tfidf_cable`
- `tfidf_cable news`
- `tfidf_case`
- `tfidf_change`
- `tfidf_click`
- `tfidf_cnn`
- `tfidf_cnncom`
- `tfidf_code`
- `tfidf_com`
- `tfidf_company`
- `tfidf_computer`
- `tfidf_contact`
- `tfidf_contenttype`
- `tfidf_contenttype textplain`
- `tfidf_credit`
- `tfidf_daily`
- `tfidf_daily 10`
- `tfidf_date`
- `tfidf_domain`
- `tfidf_dont`
- `tfidf_email`
- `tfidf_feb`
- `tfidf_feb 2008`
- `tfidf_file`
- `tfidf_free`
- `tfidf_fri`
- `tfidf_going`
- `tfidf_good`
- `tfidf_got`
- `tfidf_great`
- `tfidf_help`
- `tfidf_http`
- `tfidf_http www`
- `tfidf_im`
- `tfidf_information`
- `tfidf_intelligence`
- `tfidf_internet`
- `tfidf_john`
- `tfidf_know`
- `tfidf_language`
- `tfidf_life`
- `tfidf_like`
- `tfidf_line`
- `tfidf_list`
- `tfidf_lllp`
- `tfidf_look`
- `tfidf_lp`
- `tfidf_lp lllp`
- `tfidf_mail`
- `tfidf_mailing`
- `tfidf_mailing list`
- `tfidf_make`
- `tfidf_message`
- `tfidf_money`
- `tfidf_need`
- `tfidf_network`
- `tfidf_network lp`
- `tfidf_new`
- `tfidf_news`
- `tfidf_news network`
- `tfidf_notes`
- `tfidf_notes submissionid`
- `tfidf_number`
- `tfidf_online`
- `tfidf_order`
- `tfidf_people`
- `tfidf_perl`
- `tfidf_pm`
- `tfidf_privacy`
- `tfidf_problem`
- `tfidf_program`
- `tfidf_read`
- `tfidf_really`
- `tfidf_receive`
- `tfidf_replica`
- `tfidf_report`
- `tfidf_reports`
- `tfidf_return`
- `tfidf_science`
- `tfidf_search`
- `tfidf_security`
- `tfidf_send`
- `tfidf_sender`
- `tfidf_sender virus`
- `tfidf_settings`
- `tfidf_site`
- `tfidf_software`
- `tfidf_spam`
- `tfidf_stories`
- `tfidf_subject`
- `tfidf_submission`
- `tfidf_submission notes`
- `tfidf_submissionid`
- `tfidf_submissionid 2577781`
- `tfidf_sun`
- `tfidf_textplain`
- `tfidf_think`
- `tfidf_thu`
- `tfidf_thu 07`
- `tfidf_time`
- `tfidf_today`
- `tfidf_total`
- `tfidf_total submission`
- `tfidf_university`
- `tfidf_unsubscribe`
- `tfidf_use`
- `tfidf_used`
- `tfidf_user`
- `tfidf_using`
- `tfidf_version`
- `tfidf_videos`
- `tfidf_virus`
- `tfidf_virus total`
- `tfidf_want`
- `tfidf_watches`
- `tfidf_way`
- `tfidf_web`
- `tfidf_wed`
- `tfidf_wed 06`
- `tfidf_work`
- `tfidf_workshop`
- `tfidf_wrote`
- `tfidf_www`

### OSINT-Enhanced Features (173 features)

**Text Features** (159 features):
- `n_urls`
- `n_domains`
- `n_ips`
- `has_https`
- `has_http`
- `n_suspicious_keywords`
- `email_length`
- `n_attachments`
- `count_suspicious_tld`
- `tfidf_000`
- `tfidf_0000`
- `tfidf_0100`
- `tfidf_0500`
- `tfidf_06`
- `tfidf_06 aug`
- `tfidf_07`
- `tfidf_07 aug`
- `tfidf_08`
- `tfidf_0800`
- `tfidf_10`
- `tfidf_100`
- `tfidf_20`
- `tfidf_2002`
- `tfidf_2008`
- `tfidf_24`
- `tfidf_2577781`
- `tfidf_2577781 added`
- `tfidf________________________________________________`
- `tfidf_account`
- `tfidf_added`
- `tfidf_added submissionid`
- `tfidf_address`
- `tfidf_aug`
- `tfidf_aug 2008`
- `tfidf_best`
- `tfidf_business`
- `tfidf_cable`
- `tfidf_cable news`
- `tfidf_case`
- `tfidf_change`
- `tfidf_click`
- `tfidf_cnn`
- `tfidf_cnncom`
- `tfidf_code`
- `tfidf_com`
- `tfidf_company`
- `tfidf_computer`
- `tfidf_contact`
- `tfidf_contenttype`
- `tfidf_contenttype textplain`
- `tfidf_credit`
- `tfidf_daily`
- `tfidf_daily 10`
- `tfidf_date`
- `tfidf_domain`
- `tfidf_dont`
- `tfidf_email`
- `tfidf_feb`
- `tfidf_feb 2008`
- `tfidf_file`
- `tfidf_free`
- `tfidf_fri`
- `tfidf_going`
- `tfidf_good`
- `tfidf_got`
- `tfidf_great`
- `tfidf_help`
- `tfidf_http`
- `tfidf_http www`
- `tfidf_im`
- `tfidf_information`
- `tfidf_intelligence`
- `tfidf_internet`
- `tfidf_john`
- `tfidf_know`
- `tfidf_language`
- `tfidf_life`
- `tfidf_like`
- `tfidf_line`
- `tfidf_list`
- `tfidf_lllp`
- `tfidf_look`
- `tfidf_lp`
- `tfidf_lp lllp`
- `tfidf_mail`
- `tfidf_mailing`
- `tfidf_mailing list`
- `tfidf_make`
- `tfidf_message`
- `tfidf_money`
- `tfidf_need`
- `tfidf_network`
- `tfidf_network lp`
- `tfidf_new`
- `tfidf_news`
- `tfidf_news network`
- `tfidf_notes`
- `tfidf_notes submissionid`
- `tfidf_number`
- `tfidf_online`
- `tfidf_order`
- `tfidf_people`
- `tfidf_perl`
- `tfidf_pm`
- `tfidf_privacy`
- `tfidf_problem`
- `tfidf_program`
- `tfidf_read`
- `tfidf_really`
- `tfidf_receive`
- `tfidf_replica`
- `tfidf_report`
- `tfidf_reports`
- `tfidf_return`
- `tfidf_science`
- `tfidf_search`
- `tfidf_security`
- `tfidf_send`
- `tfidf_sender`
- `tfidf_sender virus`
- `tfidf_settings`
- `tfidf_site`
- `tfidf_software`
- `tfidf_spam`
- `tfidf_stories`
- `tfidf_subject`
- `tfidf_submission`
- `tfidf_submission notes`
- `tfidf_submissionid`
- `tfidf_submissionid 2577781`
- `tfidf_sun`
- `tfidf_textplain`
- `tfidf_think`
- `tfidf_thu`
- `tfidf_thu 07`
- `tfidf_time`
- `tfidf_today`
- `tfidf_total`
- `tfidf_total submission`
- `tfidf_university`
- `tfidf_unsubscribe`
- `tfidf_use`
- `tfidf_used`
- `tfidf_user`
- `tfidf_using`
- `tfidf_version`
- `tfidf_videos`
- `tfidf_virus`
- `tfidf_virus total`
- `tfidf_want`
- `tfidf_watches`
- `tfidf_way`
- `tfidf_web`
- `tfidf_wed`
- `tfidf_wed 06`
- `tfidf_work`
- `tfidf_workshop`
- `tfidf_wrote`
- `tfidf_www`

**OSINT Features** (61 features):
- `tfidf_00`
- `tfidf_01`
- `tfidf_05`
- `tfidf_11`
- `tfidf_12`
- `tfidf_15`
- `tfidf_2000`
- `tfidf_2001`
- `tfidf_30`
- `tfidf_alert`
- `tfidf_available`
- `tfidf_bank`
- `tfidf_cc`
- `tfidf_center`
- `tfidf_conference`
- `tfidf_corp`
- `tfidf_data`
- `tfidf_day`
- `tfidf_deal`
- `tfidf_ect`
- `tfidf_ect ect`
- `tfidf_energy`
- `tfidf_enron`
- `tfidf_following`
- `tfidf_forward`
- `tfidf_gas`
- `tfidf_group`
- `tfidf_home`
- `tfidf_hou`
- `tfidf_hou ect`
- `tfidf_let`
- `tfidf_market`
- `tfidf_million`
- `tfidf_power`
- `tfidf_price`
- `tfidf_research`
- `tfidf_said`
- `tfidf_sent`
- `tfidf_service`
- `tfidf_set`
- `tfidf_thanks`
- `tfidf_tue`
- `tfidf_vince`
- `tfidf_week`
- `tfidf_world`
- `tfidf_year`
- `tfidf_years`
- `domain_age_days`
- `host_up`
- `common_web_ports_open`
- `open_ports_count`
- `filtered_ports_count`
- `https_supported`
- `latency`
- `scan_duration`
- `alternate_ip_count`
- `asn_found`
- `host_found`
- `ip_found`
- `interesting_url`
- `has_registrar`

---

## 💡 Key Findings

1. **OSINT features improve model accuracy by 1.10%**
2. Paper reported +1.32% improvement, our result shows 1.10%
3. Both results confirm OSINT enhancement benefits

---

## 📁 Generated Files

- `comparison_table.csv` - Metrics comparison table
- `confusion_matrices.png` - Confusion matrix visualization
- `metrics_comparison.png` - Performance metrics bar chart
- `improvement_analysis.png` - Improvement analysis
- `feature_comparison.png` - Feature count vs accuracy
- `COMPARISON_REPORT.md` - This report

---

*Generated: 2026-01-11 22:56:31*
