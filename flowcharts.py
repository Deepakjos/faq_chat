import streamlit as st

def current_account_flow():
    st.header("Current Account Opening Flowchart")
    st.markdown("**Interactive guide for banks, based on RBI rules (circular).**")
    has_cc_od = st.radio(
        "Does the borrower have a CC/OD facility from the banking system?",
        ("Select an option", "Yes", "No"), key="cc_od_check"
    )
    if has_cc_od == "Yes":
        st.write("---")
        exposure = st.radio(
            "What is the aggregate exposure of the banking system to the borrower?",
            ("Select an option", "Less than Rs. 5 crore", "Rs. 5 crore or more"),
            key="exposure_check_yes"
        )
        if exposure == "Less than Rs. 5 crore":
            st.success("Any bank can open a current account, subject to undertaking from the borrower.")
        elif exposure == "Rs. 5 crore or more":
            st.info(
                "- Any bank with at least 10% of the banking system's exposure can open current accounts.\n"
                "- Other lending banks: collection accounts allowed.\n"
                "- Non-lending banks: cannot open current/collection accounts."
            )
    elif has_cc_od == "No":
        st.write("---")
        exposure = st.radio(
            "What is the aggregate exposure of the banking system to the borrower?",
            (
                "Select an option",
                "Less than Rs. 5 crore",
                "Rs. 5 crore or more but less than Rs. 50 crore",
                "Rs. 50 crore or more"
            ),
            key="exposure_check_no"
        )
        if exposure == "Less than Rs. 5 crore":
            st.success("Any bank can open a current account, subject to undertaking from the borrower.")
        elif exposure == "Rs. 5 crore or more but less than Rs. 50 crore":
            st.info("Lending banks: current accounts allowed. Non-lending banks: only collection accounts allowed.")
        elif exposure == "Rs. 50 crore or more":
            st.info(
                "- Escrow mechanism required. Only the escrow manager lending bank can open current accounts.\n"
                "- Other lending banks: collection accounts allowed.\n"
                "- Non-lending banks: cannot open any current/collection accounts."
            )
    st.markdown("---")
