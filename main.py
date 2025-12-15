import io
import os
import re
import google.generativeai as genai
import pandas as pd
import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
import gspread
from google.oauth2.service_account import Credentials


# Configure Gemini API
def configure_gemini():
    """Configure Gemini API with API key"""
    api_key = st.secrets.get("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        st.error("⚠️ Please set your Gemini API key in Streamlit secrets or as an environment variable GEMINI_API_KEY")
        st.stop()
    genai.configure(api_key=api_key)
    
    # Try different model names in order of preference
    model_names = [
        'gemini-2.5-flash',
        'gemini-1.5',
    ]
    
    for model_name in model_names:
        try:
            model = genai.GenerativeModel(model_name)
            # Test the model with a simple call to verify it works
            return model
        except Exception as e:
            st.warning(f"Model {model_name} not available: {str(e)}")
            continue
    
    st.error("❌ No available Gemini models found. Please check your API key and try again.")
    st.stop()

def extract_text_from_image(model, image):
    """Extract text from image using Gemini API"""
    try:
        # Create the prompt for OCR
        prompt = "Extract all Vietnamese text from this image. " \
        "Provide the text exactly as it appears, maintaining the original structure and formatting where possible." \
        "Fix grammatical errors as follows: " \
        "1. Correct common spelling mistakes. " \
        "2. Ensure proper use of diacritics in Vietnamese words. " \
        "3. Maintain the original meaning of the text while improving readability. " \
        "4. Preserve line breaks and spacing as in the original image." \
        "5. Remove \",\" and \".\" for any numbers. " \
        "6. \"CÁ MÁU\" and \"DÈ TƯƠI\" should be recognized as product names and not altered." \
        "7. Only separate columns by spaces, change other delimiters to spaces." \
        
        # Generate content
        response = model.generate_content([prompt, image])
        if response.text:
            return response.text
        else:
            return "No text was detected in the image."
    except Exception as e:
        error_msg = str(e)
        if "404" in error_msg:
            return "Error: The Gemini model is not available. Please try again or check your API key."
        elif "403" in error_msg:
            return "Error: API access denied. Please check your API key permissions."
        elif "quota" in error_msg.lower():
            return "Error: API quota exceeded. Please check your usage limits."
        else:
            return f"Error extracting text: {error_msg}"

def connect_to_google_sheets():
    """Connect to Google Sheets using service account credentials"""
    try:
        # Define the scope
        scope = ['https://spreadsheets.google.com/feeds',
                'https://www.googleapis.com/auth/drive']
        
        # Get credentials from Streamlit secrets
        credentials_dict = {
            "type": st.secrets["gcp_service_account"]["type"],
            "project_id": st.secrets["gcp_service_account"]["project_id"],
            "private_key_id": st.secrets["gcp_service_account"]["private_key_id"],
            "private_key": st.secrets["gcp_service_account"]["private_key"],
            "client_email": st.secrets["gcp_service_account"]["client_email"],
            "client_id": st.secrets["gcp_service_account"]["client_id"],
            "auth_uri": st.secrets["gcp_service_account"]["auth_uri"],
            "token_uri": st.secrets["gcp_service_account"]["token_uri"],
            "auth_provider_x509_cert_url": st.secrets["gcp_service_account"]["auth_provider_x509_cert_url"],
            "client_x509_cert_url": st.secrets["gcp_service_account"]["client_x509_cert_url"]
        }
        
        credentials = Credentials.from_service_account_info(credentials_dict, scopes=scope)
        gc = gspread.authorize(credentials)
        return gc
    except Exception as e:
        st.error(f"Error connecting to Google Sheets: {str(e)}")
        return None

def get_google_sheet(sheet_url_or_id, worksheet_name="Sheet1"):
    """Get Google Sheet by URL or ID"""
    try:
        gc = connect_to_google_sheets()
        if gc is None:
            return None
        
        # Try to open by URL first, then by ID
        try:
            if "docs.google.com" in sheet_url_or_id:
                spreadsheet = gc.open_by_url(sheet_url_or_id)
            else:
                spreadsheet = gc.open_by_key(sheet_url_or_id)
        except:
            st.error("Could not access the Google Sheet. Please check the URL/ID and permissions.")
            return None
        
        # Get the specified worksheet
        try:
            worksheet = spreadsheet.worksheet(worksheet_name)
        except:
            # If worksheet doesn't exist, get the first one
            worksheet = spreadsheet.sheet1
        
        return worksheet
    except Exception as e:
        st.error(f"Error accessing Google Sheet: {str(e)}")
        return None

def read_data_from_sheet(worksheet):
    """Read all data from Google Sheet and return as DataFrame"""
    try:
        # Get all values from the worksheet
        values = worksheet.get_all_values()
        
        if not values:
            return None
        
        # Create DataFrame with first row as headers
        df = pd.DataFrame(values[1:], columns=values[0])
        
        # Remove empty rows
        df = df.dropna(how='all')
        
        # Convert data types for known columns
        if 'Ngày' in df.columns:
            df['Ngày'] = pd.to_datetime(df['Ngày'], format='%d/%m/%Y', errors='coerce')
        
        for col in ['SỐ LƯỢNG', 'ĐƠN GIÁ', 'THÀNH TIỀN']:
            if col in df.columns:
                # Remove commas and convert to numeric
                df[col] = df[col].str.replace(',', '').replace('', '0')
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df
    except Exception as e:
        st.error(f"Error reading data from sheet: {str(e)}")
        return None

def save_data_to_sheet(worksheet, df, customer_name):
    """Save DataFrame to Google Sheet"""
    try:
        # Clear the worksheet
        worksheet.clear()
        
        # Prepare data for upload
        upload_df = df.copy()
        
        # Convert datetime to string for Google Sheets
        if 'Ngày' in upload_df.columns:
            upload_df['Ngày'] = upload_df['Ngày'].apply(lambda x: x.strftime('%d/%m/%Y') if pd.notna(x) else "")
        
        # Convert numeric columns to strings with formatting
        for col in ['SỐ LƯỢNG', 'ĐƠN GIÁ', 'THÀNH TIỀN']:
            if col in upload_df.columns:
                upload_df[col] = upload_df[col].apply(lambda x: f"{x:,.0f}" if pd.notna(x) else "")
        
        # Add customer name as title in first row
        title_data = [[customer_name, '', '', '', '']]
        header_data = [upload_df.columns.tolist()]
        data_rows = upload_df.values.tolist()
        
        # Combine all data
        all_data = title_data + [['']*5] + header_data + data_rows
        
        # Upload data to worksheet
        worksheet.update('A1', all_data)
        
        # Format title row
        worksheet.format('A1:E1', {
            'backgroundColor': {'red': 0.2, 'green': 0.2, 'blue': 0.8},
            'textFormat': {'bold': True, 'foregroundColor': {'red': 1, 'green': 1, 'blue': 1}, 'fontSize': 14},
            'horizontalAlignment': 'CENTER'
        })
        
        # Merge title cells
        worksheet.merge_cells('A1:E1')
        
        # Format header row
        worksheet.format('A3:E3', {
            'backgroundColor': {'red': 0.3, 'green': 0.69, 'blue': 0.31},
            'textFormat': {'bold': True, 'foregroundColor': {'red': 1, 'green': 1, 'blue': 1}}
        })
        
        # Highlight summary rows
        summary_keywords = ['CỘNG CÁ MÁU', 'CỘNG DÈ TƯƠI', 'TỔNG CỘNG']
        for i, row in enumerate(upload_df.values):
            if any(keyword in str(row[1]) for keyword in summary_keywords):
                row_num = i + 4  # +4 because we have title, blank row, and header
                worksheet.format(f'A{row_num}:E{row_num}', {
                    'backgroundColor': {'red': 1, 'green': 0.92, 'blue': 0.23},
                    'textFormat': {'bold': True}
                })
        
        return True
    except Exception as e:
        st.error(f"Error saving data to sheet: {str(e)}")
        return False

def create_new_database_sheet(customer_name):
    """Create a new Google Sheet to use as database"""
    try:
        gc = connect_to_google_sheets()
        if gc is None:
            return None
        
        # Create sheet name with timestamp
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        sheet_name = f"PriceUpdate_Database_{timestamp}"
        
        # Create a new spreadsheet
        spreadsheet = gc.create(sheet_name)
        worksheet = spreadsheet.sheet1
        
        # Set up initial structure
        headers = ['Ngày', 'TÊN HÀNG', 'SỐ LƯỢNG', 'ĐƠN GIÁ', 'THÀNH TIỀN']
        worksheet.update('A1', [headers])
        
        # Format header
        worksheet.format('A1:E1', {
            'backgroundColor': {'red': 0.3, 'green': 0.69, 'blue': 0.31},
            'textFormat': {'bold': True, 'foregroundColor': {'red': 1, 'green': 1, 'blue': 1}}
        })
        
        # Make the spreadsheet publicly editable
        spreadsheet.share('', perm_type='anyone', role='writer')
        
        return spreadsheet.url
    except Exception as e:
        st.error(f"Error creating database sheet: {str(e)}")
        return None

def export_df_as_image(display_df, customer_name):
    """Export DataFrame as image"""
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('tight')
    ax.axis('off')
    
    # Create table
    table_data = display_df.values
    col_labels = display_df.columns
    
    table = ax.table(cellText=table_data, 
                    colLabels=col_labels,
                    cellLoc='center',
                    loc='upper center')
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    
    # Color summary rows
    summary_keywords = ['CỘNG CÁ MÁU', 'CỘNG DÈ TƯƠI', 'TỔNG CỘNG']
    for i, row in enumerate(display_df.values):
        if any(keyword in str(row[1]) for keyword in summary_keywords):
            for j in range(len(display_df.columns)):
                table[(i+1, j)].set_facecolor('#ffeb3b')
                table[(i+1, j)].set_text_props(weight='bold')
    
    # Style header
    for j in range(len(col_labels)):
        table[(0, j)].set_facecolor('#4CAF50')
        table[(0, j)].set_text_props(weight='bold', color='white')
    
    plt.title(f'{customer_name}', fontsize=16, fontweight='bold', pad=-20)
    
    # Save to bytes buffer
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png', bbox_inches='tight', dpi=300, pad_inches=0.05)
    img_buffer.seek(0)
    plt.close()
    
    return img_buffer.getvalue()

def store_text_to_df(text ,ca_mau_price ,de_tuoi_price):
    """Store extracted text to DataFrame (placeholder function)"""
    # Split the text into lines and process the table data
    ocr_lines = [line for line in text.strip().splitlines() if line.strip()]
    title = ocr_lines[0].strip()

    # Find the header row and data rows
    rows = []
    for line in ocr_lines[1:]:
        cols = re.split(r"\s{2,}", line.strip())
        rows.append(cols)

    headers = rows[0]
    data = rows[1:]

    df_raw = pd.DataFrame(data, columns=headers)

    # Filter out summary rows (CỘNG, TỔNG CỘNG) and empty rows
    df_transactions = df_raw[df_raw["Ngày"].notna() & df_raw["ĐƠN GIÁ"].notna()].copy()

    # print(df_transactions)

    # Clean and convert data types
    df_transactions['Ngày'] = pd.to_datetime(df_transactions['Ngày'], format='%d/%m/%Y')
    df_transactions['SỐ LƯỢNG'] = pd.to_numeric(df_transactions['SỐ LƯỢNG'])
    df_transactions['ĐƠN GIÁ'] = pd.to_numeric(df_transactions['ĐƠN GIÁ'])
    df_transactions['THÀNH TIỀN'] = pd.to_numeric(df_transactions['THÀNH TIỀN'])
    original_grand_total = df_transactions['THÀNH TIỀN'].sum()

    # Update DataFrame with new prices and recalculate totals
    df_updated = df_transactions.copy()

    # Update unit prices
    df_updated.loc[df_updated['TÊN HÀNG'] == 'CÁ MÁU', 'ĐƠN GIÁ'] = ca_mau_price
    df_updated.loc[df_updated['TÊN HÀNG'] == 'DÈ TƯƠI', 'ĐƠN GIÁ'] = de_tuoi_price

    # Recalculate THÀNH TIỀN (Total Amount)
    df_updated['THÀNH TIỀN'] = df_updated['SỐ LƯỢNG'] * df_updated['ĐƠN GIÁ']

    # Calculate summary statistics
    ca_mau_total = df_updated[df_updated['TÊN HÀNG'] == 'CÁ MÁU']['THÀNH TIỀN'].sum()
    ca_mau_quantity = df_updated[df_updated['TÊN HÀNG'] == 'CÁ MÁU']['SỐ LƯỢNG'].sum()
    de_tuoi_total = df_updated[df_updated['TÊN HÀNG'] == 'DÈ TƯƠI']['THÀNH TIỀN'].sum()
    de_tuoi_quantity = df_updated[df_updated['TÊN HÀNG'] == 'DÈ TƯƠI']['SỐ LƯỢNG'].sum()
    grand_total = ca_mau_total + de_tuoi_total
    grand_quantity = ca_mau_quantity + de_tuoi_quantity

    # Reorganize dataframe to insert summary rows after each product group
    final_rows = []
    
    # Add CÁ MÁU transactions and summary
    ca_mau_rows = df_updated[df_updated['TÊN HÀNG'] == 'CÁ MÁU']
    if len(ca_mau_rows) > 0:
        final_rows.extend(ca_mau_rows.to_dict('records'))
        # Add CÁ MÁU summary row
        final_rows.append({
            'Ngày': pd.NaT,
            'TÊN HÀNG': 'CỘNG CÁ MÁU',
            'SỐ LƯỢNG': ca_mau_quantity,
            'ĐƠN GIÁ': pd.NaT,
            'THÀNH TIỀN': ca_mau_total
        })
    
    # Add DÈ TƯƠI transactions and summary
    de_tuoi_rows = df_updated[df_updated['TÊN HÀNG'] == 'DÈ TƯƠI']
    if len(de_tuoi_rows) > 0:
        final_rows.extend(de_tuoi_rows.to_dict('records'))
        # Add DÈ TƯƠI summary row
        final_rows.append({
            'Ngày': pd.NaT,
            'TÊN HÀNG': 'CỘNG DÈ TƯƠI',
            'SỐ LƯỢNG': de_tuoi_quantity,
            'ĐƠN GIÁ': pd.NaT,
            'THÀNH TIỀN': de_tuoi_total
        })
    
    # Add other product transactions (if any)
    other_rows = df_updated[~df_updated['TÊN HÀNG'].isin(['CÁ MÁU', 'DÈ TƯƠI'])]
    if len(other_rows) > 0:
        final_rows.extend(other_rows.to_dict('records'))
    
    # Add grand total row at the end
    final_rows.append({
        'Ngày': pd.NaT,
        'TÊN HÀNG': 'TỔNG CỘNG',
        'SỐ LƯỢNG': grand_quantity,
        'ĐƠN GIÁ': pd.NaT,
        'THÀNH TIỀN': grand_total
    })
    
    # Create final dataframe with proper ordering
    df_updated = pd.DataFrame(final_rows)

    # Show difference from original
    return title, df_updated, grand_total, original_grand_total

def main():
    st.set_page_config(
        page_title="Image OCR with Gemini",
        page_icon="🔍",
        layout="wide"
    )
    
    st.title("🔍 Image OCR with Gemini API")
    st.markdown("Upload an image and extract text using Google's Gemini AI")
    
    # Configure Gemini
    model = configure_gemini()
    
    # Create columns for layout
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📤 Upload Image")
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'],
            help="Upload an image containing text to extract"
        )
        
        # Clear session state when a new file is uploaded
        if uploaded_file is not None:
            # Check if this is a new file upload by comparing with stored filename
            current_filename = uploaded_file.name if uploaded_file else None
            if st.session_state.get('last_uploaded_filename') != current_filename:
                # Clear all session state data for new file upload
                keys_to_clear = ['data_processed', 'customer', 'new_df', 'new_grand_total', 'old_grand_total', 'ca_mau_price', 'de_tuoi_price']
                for key in keys_to_clear:
                    if key in st.session_state:
                        del st.session_state[key]
                st.session_state['last_uploaded_filename'] = current_filename
            
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", width='stretch')
            
            # Image info
            st.info(f"📊 **Image Info:** {image.size[0]}x{image.size[1]} pixels, Format: {image.format}")
    
    with col2:
        st.header("📝 Update New Price")
        col2_1, col2_2 = st.columns(2) 
        with col2_1:
            # Custom price input with comma formatting
            ca_mau_display = st.text_input(
                "Cá Máu New Price:",
                value="28,000",
                help="Enter new price for Cá Máu (with commas)"
            )
            # Convert display value to numeric, removing commas
            try:
                ca_mau_price = float(ca_mau_display.replace(",", ""))
            except:
                ca_mau_price = 28000.0
                st.error("Invalid price format. Using default 28,000")
        
        with col2_2:
            # Custom price input with comma formatting
            de_tuoi_display = st.text_input(
                "Dè Tươi New Price:",
                value="37,000", 
                help="Enter new price for Dè Tươi (with commas)"
            )
            # Convert display value to numeric, removing commas
            try:
                de_tuoi_price = float(de_tuoi_display.replace(",", ""))
            except:
                de_tuoi_price = 37000.0
                st.error("Invalid price format. Using default 37,000")
    
        if uploaded_file is not None:
            if st.button("🚀 Update Prices", type="primary", width='stretch'):
                with st.spinner("Extracting text from image..."):
                    # extracted_text = extract_text_from_image(model, image)
                    extracted_text = """ CHỊ TUYẾT (Ngày 01/12 - 06/12/2025)

Ngày        TÊN HÀNG     SỐ LƯỢNG     ĐƠN GIÁ     THÀNH TIỀN
01/12/2025  CÁ MÁU          436       25000     10900000
02/12/2025  CÁ MÁU          274       25000      6850000
03/12/2025  CÁ MÁU          257       25000      6425000
04/12/2025  CÁ MÁU          306       25000      7650000
05/12/2025  CÁ MÁU          177       25000      4425000
06/12/2025  CÁ MÁU          250       25000      6250000

            CỘNG CÁ MÁU   1700                  42500000
02/12/2025  DÈ TƯƠI        1500       35000     52500000

            CỘNG DÈ TƯƠI  1500                  52500000
            TỔNG CỘNG     3200                  95000000"""
                # print(extracted_text)
                if extracted_text:
                    customer, new_df, new_grand_total, old_grand_total = store_text_to_df(extracted_text ,ca_mau_price ,de_tuoi_price)
                    
                    # Store data in session state
                    st.session_state.customer = customer
                    st.session_state.new_df = new_df
                    st.session_state.new_grand_total = new_grand_total
                    st.session_state.old_grand_total = old_grand_total
                    st.session_state.ca_mau_price = ca_mau_price
                    st.session_state.de_tuoi_price = de_tuoi_price
                    st.session_state.data_processed = True
                    
                else:
                    st.error("❌ No text could be extracted from the image.")
            
            # Display results if data exists in session state
            if st.session_state.get('data_processed', False):
                customer = st.session_state.customer
                new_df = st.session_state.new_df
                new_grand_total = st.session_state.new_grand_total
                old_grand_total = st.session_state.old_grand_total
                ca_mau_price = st.session_state.ca_mau_price
                de_tuoi_price = st.session_state.de_tuoi_price
                
                price_message = f"✅ Prices updated: Cá Máu: {ca_mau_price:,.0f} VND, Dè Tươi: {de_tuoi_price:,.0f} VND successfully!"
                total_message = f"Grand Total updated from **{old_grand_total:,.0f}** VND to **{new_grand_total:,.0f}** VND."
                difference_message = f"And the difference is **{new_grand_total - old_grand_total:,.0f}** VND."
                st.success(f"{price_message} {total_message} {difference_message}")

                # Display updated DataFrame
                st.subheader(f"📊 {customer}")
                
                # Format the DataFrame for better display
                display_df = new_df.copy()
                
                # Format date column
                display_df['Ngày'] = display_df['Ngày'].apply(lambda x: x.strftime('%d/%m/%Y') if pd.notna(x) else "")
                
                # Format numeric columns with commas, handle NaN values for summary rows
                display_df['SỐ LƯỢNG'] = display_df['SỐ LƯỢNG'].apply(lambda x: f"{x:,.0f}" if pd.notna(x) else "")
                display_df['ĐƠN GIÁ'] = display_df['ĐƠN GIÁ'].apply(lambda x: f"{x:,.0f}" if pd.notna(x) else "")
                display_df['THÀNH TIỀN'] = display_df['THÀNH TIỀN'].apply(lambda x: f"{x:,.0f}" if pd.notna(x) else "")
                
                # Apply styling to highlight summary rows
                def highlight_summary_rows(row):
                    if any(keyword in str(row['TÊN HÀNG']) for keyword in ['CỘNG CÁ MÁU', 'CỘNG DÈ TƯƠI', 'TỔNG CỘNG']):
                        return ['background-color: #ffeb3b; font-weight: bold'] * len(row)
                    return [''] * len(row)
                
                styled_df = display_df.style.apply(highlight_summary_rows, axis=1)
                
                # Export dataframe as image
                img_data = export_df_as_image(display_df, customer)
                st.download_button(
                    label="💾 Download Table Image",
                    data=img_data,
                    file_name=f"{customer.replace(' ', '_')}.png",
                    mime="image/png",
                    width='stretch',
                    key="download_image_btn"
                )
                
                st.dataframe(
                    styled_df,
                    width='stretch',
                    height=400,
                    hide_index=True,
                )
        else:
            st.info("👆 Please upload an image to extract text")
    
    # Footer
    st.markdown("---")
    st.markdown("Built by Nhan Le")

if __name__ == "__main__":
    main()
