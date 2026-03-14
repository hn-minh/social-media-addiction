import streamlit as st
import requests
from dotenv import load_dotenv
import os

load_dotenv(override=True)

API_PREDICT_URL = os.getenv("API_PREDICT_URL")
API_COLLECT_URL = os.getenv("API_COLLECT_URL")

st.set_page_config(
    page_title="AI Dự đoán Nghiện MXH", 
    page_icon="📱", 
    layout="centered"
)

st.title("📱 Hệ Thống AI Đánh Giá Mức Độ Nghiện Mạng Xã Hội")
st.markdown("""
    Chào mừng bạn đến với hệ thống đánh giá tự động. Vui lòng nhập thông tin thói quen 
    sử dụng mạng xã hội của bạn, AI của chúng tôi sẽ phân tích và đưa ra kết quả.
""")

st.divider()

# Khởi tạo session state để lưu kết quả dự đoán (phục vụ cho form feedback)
if "prediction_result" not in st.session_state:
    st.session_state.prediction_result = None
if "user_payload" not in st.session_state:
    st.session_state.user_payload = None

# --- PHẦN 1: FORM NHẬP LIỆU ---
with st.form("user_input_form"):
    st.subheader("📊 Thông tin & Hành vi của bạn")
    
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Tuổi", min_value=10, max_value=100, value=20)
        gender = st.selectbox("Giới tính", ["Male", "Female", "Other"])
        academic_level = st.selectbox("Trình độ học vấn", ["High School", "Undergraduate", "Postgraduate"])
        country = st.text_input("Quốc gia (VD: Vietnam, USA, Japan)", value="Vietnam")
        relationship = st.selectbox("Tình trạng mối quan hệ", ["Single", "In a relationship", "Married", "Divorced"])
        
    with col2:
        platform = st.selectbox("Nền tảng sử dụng nhiều nhất", ["Facebook", "TikTok", "Instagram", "Twitter", "YouTube", "Other"])
        usage_hours = st.slider("Số giờ dùng MXH trung bình/ngày", 0.0, 24.0, 3.0, step=0.5)
        sleep_hours = st.slider("Số giờ ngủ trung bình/đêm", 0.0, 16.0, 7.0, step=0.5)
        mental_health = st.slider("Điểm sức khỏe tinh thần (1-Tệ nhất, 10-Tốt nhất)", 1, 10, 5)
        conflicts = st.slider("Số lần xung đột do MXH (trong tháng)", 0, 30, 0)
        
    affects_academic = st.checkbox("Mạng xã hội có ảnh hưởng tiêu cực đến kết quả học tập/công việc của bạn không?")

    # Nút submit form
    submit_btn = st.form_submit_button("🚀 Phân Tích & Dự Đoán", type="primary", use_container_width=True)

# --- PHẦN 2: XỬ LÝ GỌI API DỰ ĐOÁN ---
if submit_btn:
    # Gói dữ liệu thành JSON khớp với Schema Pydantic
    payload = {
        "Age": age,
        "Gender": gender,
        "Academic_Level": academic_level,
        "Country": country,
        "Avg_Daily_Usage_Hours": usage_hours,
        "Most_Used_Platform": platform,
        "Affects_Academic_Performance": affects_academic,
        "Sleep_Hours_Per_Night": sleep_hours,
        "Mental_Health_Score": mental_health,
        "Relationship_Status": relationship,
        "Conflicts_Over_Social_Media": conflicts
    }
    
    with st.spinner("AI đang xử lý dữ liệu của bạn..."):
        try:
            response = requests.post(API_PREDICT_URL, json=payload)
            if response.status_code == 200:
                result = response.json()
                # Lưu vào state để hiển thị ở phần feedback
                st.session_state.prediction_result = result
                st.session_state.user_payload = payload
            else:
                st.error(f"Lỗi từ server: {response.text}")
        except requests.exceptions.ConnectionError:
            st.error("Không thể kết nối đến Backend API. Vui lòng kiểm tra xem FastAPI đã chạy chưa!")

# --- PHẦN 3: HIỂN THỊ KẾT QUẢ VÀ THU THẬP DỮ LIỆU (FLYWHEEL) ---
if st.session_state.prediction_result:
    res = st.session_state.prediction_result
    score = res["predicted_addiction_score"]
    percentile = res["better_than_percentage"]
    
    st.success("✅ Phân tích hoàn tất!")
    
    # Giao diện hiển thị kết quả
    col_res1, col_res2 = st.columns(2)
    with col_res1:
        st.metric(label="Mức độ nghiện của bạn (Càng cao càng nặng)", value=f"{score} / 10")
    with col_res2:
        st.metric(label="So với những người khác", value=f"Tốt hơn {percentile}%", delta=f"{percentile}%", delta_color="normal")
            
    st.divider()
    
    # Form thu thập phản hồi người dùng (Đẩy vào /collect)
    st.subheader("🔄 Bạn có đồng ý với kết quả này không?")
    st.write("Hãy cung cấp mức độ thực tế của bạn để giúp AI của chúng tôi học hỏi và thông minh hơn vào ngày mai!")
    
    with st.form("feedback_form"):
        actual_score = st.slider("Mức độ nghiện thực tế của bạn (0-5)", 0, 5, score)
        feedback_submit = st.form_submit_button("Gửi phản hồi đóng góp data")
        
        if feedback_submit:
            # Lấy lại payload cũ và thêm nhãn thực tế vào
            collect_payload = st.session_state.user_payload.copy()
            collect_payload["Addicted_Score"] = actual_score
            
            try:
                res_collect = requests.post(API_COLLECT_URL, json=collect_payload)
                if res_collect.status_code == 200:
                    st.balloons()
                    st.success(f"Cảm ơn bạn! Dữ liệu đã được ghi nhận với mã: {res_collect.json()['student_id']}")
                    # Reset state
                    st.session_state.prediction_result = None
                else:
                    st.error("Có lỗi xảy ra khi lưu dữ liệu.")
            except Exception as e:
                st.error(f"Lỗi kết nối: {e}")