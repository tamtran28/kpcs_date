import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
from datetime import datetime

# --- CẤU HÌNH TRANG WEB ---
st.set_page_config(layout="wide", page_title="Hệ thống Báo cáo KPCS Tự động")
st.title("📊 Hệ thống Báo cáo Tự động")

# ==============================================================================
# PHẦN 1: CÁC HÀM LOGIC (Không thay đổi)
# ==============================================================================

def calculate_summary_metrics(dataframe, groupby_cols, year_start_date, report_start_date, report_end_date):
    """
    Hàm tính toán các chỉ số chính.
    Lưu ý: tên biến 'quarter_start_date' và 'quarter_end_date' được giữ lại trong code nội bộ
    của hàm để tránh thay đổi lớn, nhưng chúng thực chất đại diện cho ngày bắt đầu và kết thúc của kỳ báo cáo do người dùng chọn.
    """
    if not isinstance(groupby_cols, list): raise TypeError("groupby_cols phải là một danh sách (list)")
    
    # Đổi tên biến để code dễ đọc hơn
    quarter_start_date = report_start_date
    quarter_end_date = report_end_date

    def agg(data_filtered, cols):
        if data_filtered.empty: return 0 if not cols else pd.Series(dtype=int)
        if not cols: return len(data_filtered)
        return data_filtered.groupby(cols).size()

    ton_dau_ky = agg(dataframe[(dataframe['Ngày, tháng, năm ban hành (mm/dd/yyyy)'] < quarter_start_date) & ((dataframe['NGÀY HOÀN TẤT KPCS (mm/dd/yyyy)'].isnull()) | (dataframe['NGÀY HOÀN TẤT KPCS (mm/dd/yyyy)'] >= quarter_start_date))], groupby_cols)
    phat_sinh_ky = agg(dataframe[(dataframe['Ngày, tháng, năm ban hành (mm/dd/yyyy)'] >= quarter_start_date) & (dataframe['Ngày, tháng, năm ban hành (mm/dd/yyyy)'] <= quarter_end_date)], groupby_cols)
    khac_phuc_ky = agg(dataframe[(dataframe['NGÀY HOÀN TẤT KPCS (mm/dd/yyyy)'] >= quarter_start_date) & (dataframe['NGÀY HOÀN TẤT KPCS (mm/dd/yyyy)'] <= quarter_end_date)], groupby_cols)
    phat_sinh_nam = agg(dataframe[(dataframe['Ngày, tháng, năm ban hành (mm/dd/yyyy)'] >= year_start_date) & (dataframe['Ngày, tháng, năm ban hành (mm/dd/yyyy)'] <= quarter_end_date)], groupby_cols)
    khac_phuc_nam = agg(dataframe[(dataframe['NGÀY HOÀN TẤT KPCS (mm/dd/yyyy)'] >= year_start_date) & (dataframe['NGÀY HOÀN TẤT KPCS (mm/dd/yyyy)'] <= quarter_end_date)], groupby_cols)
    ton_dau_nam = agg(dataframe[(dataframe['Ngày, tháng, năm ban hành (mm/dd/yyyy)'] < year_start_date) & ((dataframe['NGÀY HOÀN TẤT KPCS (mm/dd/yyyy)'].isnull()) | (dataframe['NGÀY HOÀN TẤT KPCS (mm/dd/yyyy)'] >= year_start_date))], groupby_cols)
    
    if not groupby_cols:
        summary = pd.DataFrame({'Tồn đầu kỳ': [ton_dau_ky], 'Phát sinh kỳ': [phat_sinh_ky], 'Khắc phục kỳ': [khac_phuc_ky], 'Tồn đầu năm': [ton_dau_nam], 'Phát sinh năm': [phat_sinh_nam], 'Khắc phục năm': [khac_phuc_nam]})
    else:
        summary = pd.DataFrame({'Tồn đầu kỳ': ton_dau_ky, 'Phát sinh kỳ': phat_sinh_ky, 'Khắc phục kỳ': khac_phuc_ky, 'Tồn đầu năm': ton_dau_nam, 'Phát sinh năm': phat_sinh_nam, 'Khắc phục năm': khac_phuc_nam}).fillna(0).astype(int)
    
    summary['Tồn cuối kỳ'] = summary['Tồn đầu kỳ'] + summary['Phát sinh kỳ'] - summary['Khắc phục kỳ']
    df_actually_outstanding = dataframe[(dataframe['Ngày, tháng, năm ban hành (mm/dd/yyyy)'] <= quarter_end_date) & ((dataframe['NGÀY HOÀN TẤT KPCS (mm/dd/yyyy)'].isnull()) | (dataframe['NGÀY HOÀN TẤT KPCS (mm/dd/yyyy)'] > quarter_end_date))]
    qua_han_khac_phuc = agg(df_actually_outstanding[df_actually_outstanding['Thời hạn hoàn thành (mm/dd/yyyy)'] < quarter_end_date], groupby_cols)
    qua_han_tren_1_nam = agg(df_actually_outstanding[df_actually_outstanding['Thời hạn hoàn thành (mm/dd/yyyy)'] < (quarter_end_date - pd.DateOffset(years=1))], groupby_cols)
    summary['Quá hạn khắc phục'] = qua_han_khac_phuc; summary['Trong đó quá hạn trên 1 năm'] = qua_han_tren_1_nam
    summary = summary.fillna(0).astype(int); denominator = summary['Phát sinh năm'] + summary['Tồn đầu năm']
    summary['Tỷ lệ chưa KP đến cuối kỳ'] = (summary['Tồn cuối kỳ'] / denominator).replace([np.inf, -np.inf], 0).fillna(0)
    
    final_cols_order = ['Tồn đầu năm', 'Phát sinh năm', 'Khắc phục năm', 'Tồn đầu kỳ', 'Phát sinh kỳ', 'Khắc phục kỳ', 'Tồn cuối kỳ', 'Quá hạn khắc phục', 'Trong đó quá hạn trên 1 năm', 'Tỷ lệ chưa KP đến cuối kỳ']
    return summary.reindex(columns=final_cols_order, fill_value=0)

def create_summary_table(dataframe, groupby_col, dates):
    summary = calculate_summary_metrics(dataframe, [groupby_col], **dates)
    if not summary.empty:
        total_row = pd.DataFrame(summary.sum(numeric_only=True)).T; total_row.index = ['TỔNG CỘNG']
        total_denom = total_row.at['TỔNG CỘNG', 'Phát sinh năm'] + total_row.at['TỔNG CỘNG', 'Tồn đầu năm']
        total_row['Tỷ lệ chưa KP đến cuối kỳ'] = (total_row.at['TỔNG CỘNG', 'Tồn cuối kỳ'] / total_denom) if total_denom != 0 else 0
        summary = pd.concat([summary, total_row])
    return summary

def create_top_n_table(dataframe, n, group_by_col, dates):
    if group_by_col not in dataframe.columns: return pd.DataFrame()
    full_summary = calculate_summary_metrics(dataframe, [group_by_col], **dates)
    top_n = full_summary.sort_values(by='Quá hạn khắc phục', ascending=False).head(n)
    total_row = pd.DataFrame(full_summary.sum(numeric_only=True)).T; total_row.index = ['TỔNG CỘNG CỦA NHÓM']
    total_denom = total_row.at['TỔNG CỘNG CỦA NHÓM', 'Phát sinh năm'] + total_row.at['TỔNG CỘNG CỦA NHÓM', 'Tồn đầu năm']
    total_row['Tỷ lệ chưa KP đến cuối kỳ'] = (total_row.at['TỔNG CỘNG CỦA NHÓM', 'Tồn cuối kỳ'] / total_denom) if total_denom != 0 else 0
    return pd.concat([top_n, total_row])

def create_hierarchical_table(dataframe, parent_col, child_col, dates):
    cols_order = ['Tên Đơn vị', 'Tồn đầu năm', 'Phát sinh năm', 'Khắc phục năm', 'Tồn đầu kỳ', 'Phát sinh kỳ', 'Khắc phục kỳ', 'Tồn cuối kỳ', 'Quá hạn khắc phục', 'Trong đó quá hạn trên 1 năm', 'Tỷ lệ chưa KP đến cuối kỳ']
    if dataframe.empty or parent_col not in dataframe.columns or child_col not in dataframe.columns:
        return pd.DataFrame(columns=cols_order)
    summary_child = calculate_summary_metrics(dataframe, [child_col], **dates)
    parent_mapping = dataframe[[child_col, parent_col]].drop_duplicates()
    summary_with_parent = pd.merge(summary_child.reset_index().rename(columns={'index': child_col}), parent_mapping, on=child_col, how='left')
    final_report_rows = []
    unique_parents = sorted(dataframe[parent_col].dropna().unique())
    for parent_name in unique_parents:
        children_df = summary_with_parent[summary_with_parent[parent_col] == parent_name]
        if children_df.empty: continue
        numeric_cols = children_df.select_dtypes(include=np.number).columns
        parent_row_sum = children_df[numeric_cols].sum().to_frame().T; parent_row_sum['Tên Đơn vị'] = f"**Cộng {parent_name}**"; final_report_rows.append(parent_row_sum)
        children_to_append = children_df.rename(columns={child_col: 'Tên Đơn vị'}); children_to_append['Tên Đơn vị'] = "  •  " + children_to_append['Tên Đơn vị'].astype(str); final_report_rows.append(children_to_append)
    if not final_report_rows: return pd.DataFrame(columns=cols_order)
    full_report_df = pd.concat(final_report_rows, ignore_index=True)
    grand_total_row = calculate_summary_metrics(dataframe, [], **dates); grand_total_row['Tên Đơn vị'] = '**TỔNG CỘNG TOÀN BỘ**'
    full_report_df = pd.concat([full_report_df, grand_total_row], ignore_index=True)
    return full_report_df.reindex(columns=cols_order).fillna(0)

def create_overdue_hierarchical_report(dataframe, parent_col, child_col, dates):
    q_end = dates['report_end_date'] # Sử dụng report_end_date
    if dataframe.empty or parent_col not in dataframe.columns or child_col not in dataframe.columns:
        return pd.DataFrame()
    df_outstanding = dataframe[(dataframe['Ngày, tháng, năm ban hành (mm/dd/yyyy)'] <= q_end) & ((dataframe['NGÀY HOÀN TẤT KPCS (mm/dd/yyyy)'].isnull()) | (dataframe['NGÀY HOÀN TẤT KPCS (mm/dd/yyyy)'] > q_end))].copy()
    if df_outstanding.empty:
        st.warning(f"Không có kiến nghị tồn đọng cho nhóm báo cáo này.")
        return pd.DataFrame()
    df_overdue = df_outstanding[df_outstanding['Thời hạn hoàn thành (mm/dd/yyyy)'] < q_end].copy()
    
    summary_child = calculate_summary_metrics(dataframe, [child_col], **dates)
    
    overdue_breakdown_child = pd.DataFrame()
    labels = ['Dưới 3 tháng', 'Từ 3-6 tháng', 'Từ 6-9 tháng', 'Từ 9-12 tháng', 'Trên 1 năm']
    if not df_overdue.empty:
        df_overdue['Số ngày quá hạn'] = (q_end - df_overdue['Thời hạn hoàn thành (mm/dd/yyyy)']).dt.days
        bins = [-np.inf, 90, 180, 270, 365, np.inf]
        df_overdue['Nhóm quá hạn'] = pd.cut(df_overdue['Số ngày quá hạn'], bins=bins, labels=labels, right=False)
        overdue_breakdown_child = pd.crosstab(df_overdue[child_col], df_overdue['Nhóm quá hạn'])

    summary_child_reset = summary_child.reset_index().rename(columns={'index': child_col})
    overdue_breakdown_reset = overdue_breakdown_child.reset_index()
    summary_child_full = pd.merge(summary_child_reset, overdue_breakdown_reset, on=child_col, how='left')
    
    parent_mapping = dataframe[[child_col, parent_col]].drop_duplicates()
    summary_child_with_parent = pd.merge(summary_child_full, parent_mapping, on=child_col, how='left')

    final_report_rows = []
    unique_parents = sorted(dataframe[parent_col].dropna().unique())
    for parent_name in unique_parents:
        if 'tổng' in str(parent_name).lower(): continue
        children_df = summary_child_with_parent[summary_child_with_parent[parent_col] == parent_name]
        if children_df.empty: continue
        
        numeric_cols = children_df.select_dtypes(include=np.number).columns
        parent_row_sum = children_df[numeric_cols].sum().to_frame().T
        parent_row_sum['Tên Đơn vị'] = f"**{parent_name}**"
        final_report_rows.append(parent_row_sum)
        
        children_to_append = children_df.rename(columns={child_col: 'Tên Đơn vị'})
        children_to_append['Tên Đơn vị'] = "  • " + children_to_append['Tên Đơn vị']
        final_report_rows.append(children_to_append)
        
    if not final_report_rows: return pd.DataFrame()

    final_df = pd.concat(final_report_rows, ignore_index=True)
    
    grand_total_metrics = calculate_summary_metrics(dataframe, [], **dates)
    grand_total_overdue = pd.DataFrame()
    if not df_overdue.empty:
        grand_total_overdue = df_overdue['Nhóm quá hạn'].value_counts().to_frame().T
    grand_total_row = pd.concat([grand_total_metrics, grand_total_overdue], axis=1)
    grand_total_row['Tên Đơn vị'] = '**TỔNG CỘNG TOÀN BỘ**'
    
    final_df = pd.concat([final_df, grand_total_row])
    
    final_cols_order = ['Tên Đơn vị', 'Tồn đầu năm', 'Phát sinh năm', 'Khắc phục năm', 'Tồn đầu kỳ', 'Phát sinh kỳ', 'Khắc phục kỳ', 'Tồn cuối kỳ', 'Quá hạn khắc phục', 'Trong đó quá hạn trên 1 năm', 'Tỷ lệ chưa KP đến cuối kỳ'] + labels
    final_df = final_df.reindex(columns=final_cols_order, fill_value=0)
    numeric_cols = final_df.columns.drop('Tên Đơn vị')
    final_df[numeric_cols] = final_df[numeric_cols].fillna(0).astype(int)
    
    return final_df

def format_excel_sheet(writer, df_to_write, sheet_name, index=True):
    # Đổi tên cột từ "Quý" sang "Kỳ" trong file Excel để phù hợp với ngữ cảnh mới
    df_to_write.columns = [col.replace('Quý', 'kỳ').replace('quý', 'kỳ') for col in df_to_write.columns]
    
    df_to_write.to_excel(writer, sheet_name=sheet_name, index=index)
    workbook = writer.book; worksheet = writer.sheets[sheet_name]
    border_format = workbook.add_format({'border': 1, 'valign': 'vcenter', 'align': 'left'})
    worksheet.conditional_format(0, 0, len(df_to_write), len(df_to_write.columns) + (1 if index else 0) -1, {'type': 'no_blanks', 'format': border_format})
    for idx, col in enumerate(df_to_write.columns):
        series = df_to_write[col]; max_len = max((series.astype(str).map(len).max(), len(str(series.name)))) + 3
        worksheet.set_column(idx + (1 if index else 0), idx + (1 if index else 0), max_len)
    if index:
        max_len_idx = max(df_to_write.index.astype(str).map(len).max(), len(str(df_to_write.index.name))) + 3
        worksheet.set_column(0, 0, max_len_idx)

# ==============================================================================
# PHẦN 2: GIAO DIỆN VÀ LUỒNG THỰC THI CỦA STREAMLIT
# ==============================================================================

with st.sidebar:
    st.header("⚙️ Tùy chọn báo cáo")
    
    # --- THAY ĐỔI BẮT ĐẦU TỪ ĐÂY ---
    # Bỏ chọn Năm và Quý, thay bằng chọn khoảng thời gian
    
    # Sử dụng 2 cột để đặt widget chọn ngày cạnh nhau cho gọn
    c1, c2 = st.columns(2)
    with c1:
        start_date = st.date_input("Từ ngày", datetime(datetime.now().year, 1, 1))
    with c2:
        end_date = st.date_input("Đến ngày", datetime.now())

    # --- KẾT THÚC THAY ĐỔI ---
    
    uploaded_file = st.file_uploader("📂 Tải lên file Excel dữ liệu thô", type=["xlsx", "xls"])

if uploaded_file is not None:
    # Thêm kiểm tra logic: ngày bắt đầu không được lớn hơn ngày kết thúc
    if start_date > end_date:
        st.error("Lỗi: 'Từ ngày' không được lớn hơn 'Đến ngày'. Vui lòng chọn lại.")
    else:
        st.success(f"✅ Đã tải lên thành công file: **{uploaded_file.name}**")
        st.info(f"Kỳ báo cáo được chọn: từ **{start_date.strftime('%d/%m/%Y')}** đến **{end_date.strftime('%d/%m/%Y')}**")

        @st.cache_data
        def load_data(file):
            df = pd.read_excel(file)
            date_cols = ['Ngày, tháng, năm ban hành (mm/dd/yyyy)', 'NGÀY HOÀN TẤT KPCS (mm/dd/yyyy)', 'Thời hạn hoàn thành (mm/dd/yyyy)']
            for col in date_cols:
                if col in df.columns: df[col] = pd.to_datetime(df[col], errors='coerce')
            return df

        df_raw = load_data(uploaded_file)
        st.write("Xem trước 5 dòng dữ liệu đầu tiên:")
        st.dataframe(df_raw.head())

        df = df_raw.copy()

        # --- THAY ĐỔI CÁCH TẠO BIẾN DATES ---
        # Chuyển đổi ngày tháng từ st.date_input thành pd.Timestamp
        start_date_ts = pd.to_datetime(start_date)
        end_date_ts = pd.to_datetime(end_date)
        
        # 'year_start_date' sẽ là ngày đầu tiên của năm của `end_date`
        # để đảm bảo các chỉ số "Lũy kế từ đầu năm" được tính đúng
        year_of_end_date = end_date_ts.year
        
        dates = {
            'year_start_date': pd.to_datetime(f'{year_of_end_date}-01-01'),
            'report_start_date': start_date_ts, # Ngày bắt đầu kỳ báo cáo
            'report_end_date': end_date_ts      # Ngày kết thúc kỳ báo cáo
        }
        # --- KẾT THÚC THAY ĐỔI ---

        def clean_string(x):
            if isinstance(x, str): return x.strip()
            return '' if pd.isna(x) else str(x)
        
        PARENT_COL = 'SUM (THEO Khối, KV, ĐVKD, Hội sở, Ban Dự Án QLTS)'
        CHILD_COL = 'Đơn vị thực hiện KPCS trong quý'
        text_cols = [CHILD_COL, PARENT_COL, 'ĐVKD, AMC, Hội sở (Nhập ĐVKD hoặc Hội sở hoặc AMC)']
        for col in text_cols:
            if col in df.columns: df[col] = df[col].apply(clean_string)
        
        if PARENT_COL in df.columns:
            df = df[~df[PARENT_COL].str.lower().str.contains('tổng cộng|tổng', na=False)]
        if CHILD_COL in df.columns:
            # Đổi tên cột để logic phía sau không bị ảnh hưởng
            df.rename(columns={'Đơn vị thực hiện KPCS trong quý': 'Đơn vị thực hiện KPCS trong kỳ'}, inplace=True)
            CHILD_COL = 'Đơn vị thực hiện KPCS trong kỳ' # Cập nhật lại tên biến
            df = df[~df[CHILD_COL].str.lower().str.contains('tổng cộng|tổng', na=False)]
                
        df['Nhom_Don_Vi'] = np.where(df['ĐVKD, AMC, Hội sở (Nhập ĐVKD hoặc Hội sở hoặc AMC)'] == 'Hội sở', 'Hội sở', 'ĐVKD, AMC')
        df_hoiso = df[df['Nhom_Don_Vi'] == 'Hội sở'].copy()
        df_dvdk_amc = df[df['Nhom_Don_Vi'] == 'ĐVKD, AMC'].copy()

        st.markdown("---"); st.header("Chọn Loại Báo Cáo Để Tạo")
        col1, col2 = st.columns(2)

        with col1:
            if st.button("🚀 Tạo Báo cáo Tổng hợp (1-7)"):
                with st.spinner("⏳ Đang xử lý và tạo 7 báo cáo..."):
                    output_stream = BytesIO()
                    with pd.ExcelWriter(output_stream, engine='xlsxwriter') as writer:
                        format_excel_sheet(writer, create_summary_table(df, 'Nhom_Don_Vi', dates), "1_TH_ToanHang")
                        format_excel_sheet(writer, create_summary_table(df_hoiso, PARENT_COL, dates), "2_TH_HoiSo")
                        format_excel_sheet(writer, create_top_n_table(df_hoiso, 5, PARENT_COL, dates), "3_Top5_HoiSo")
                        format_excel_sheet(writer, create_hierarchical_table(df_hoiso, PARENT_COL, CHILD_COL, dates), "4_PhanCap_HoiSo", index=False)
                        format_excel_sheet(writer, create_summary_table(df_dvdk_amc, PARENT_COL, dates), "5_TH_DVDK_KhuVuc")
                        format_excel_sheet(writer, create_top_n_table(df_dvdk_amc, 10, CHILD_COL, dates), "6_Top10_DVDK")
                        format_excel_sheet(writer, create_hierarchical_table(df_dvdk_amc, PARENT_COL, CHILD_COL, dates), "7_ChiTiet_DVDK", index=False)
                    excel_data = output_stream.getvalue()
                st.success("🎉 Đã tạo xong file Excel Tổng hợp!")
                
                # Cập nhật tên file download
                file_name_str = f"Tong_hop_Bao_cao_KPCS_{start_date.strftime('%Y%m%d')}_den_{end_date.strftime('%Y%m%d')}.xlsx"
                st.download_button(label="📥 Tải xuống File Tổng hợp", data=excel_data, file_name=file_name_str, mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
                
        with col2:
            if st.button("📊 Tạo BC Quá hạn Chi tiết (8 & 9)"):
                with st.spinner("⏳ Đang xử lý và tạo Báo cáo quá hạn chi tiết..."):
                    df8 = create_overdue_hierarchical_report(df_hoiso, PARENT_COL, CHILD_COL, dates)
                    df9 = create_overdue_hierarchical_report(df_dvdk_amc, PARENT_COL, CHILD_COL, dates)
                    
                    output_stream_overdue = BytesIO()
                    with pd.ExcelWriter(output_stream_overdue, engine='xlsxwriter') as writer:
                        if not df8.empty:
                            format_excel_sheet(writer, df8, "8_BC_QuaHan_Pcap_HoiSo", index=False)
                        if not df9.empty:
                            format_excel_sheet(writer, df9, "9_BC_QuaHan_Pcap_DVDK", index=False)
                    
                    excel_data_overdue = output_stream_overdue.getvalue()
                st.success("🎉 Đã tạo xong file Excel Quá hạn chi tiết!")
                
                # Cập nhật tên file download
                file_name_str_overdue = f"BC_QuaHan_ChiTiet_{start_date.strftime('%Y%m%d')}_den_{end_date.strftime('%Y%m%d')}.xlsx"
                st.download_button(label="📥 Tải xuống File Quá hạn Chi tiết", data=excel_data_overdue, file_name=file_name_str_overdue, mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

else:
    st.info("💡 Vui lòng tải lên file Excel chứa dữ liệu thô để bắt đầu.")
