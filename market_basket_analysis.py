import pandas as pd 
from flask import Blueprint, render_template, request, redirect, url_for, flash
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

bp = Blueprint("market_basket_analysis", __name__)
te = TransactionEncoder()

@bp.route("/market_basket_analysis", methods=["GET", "POST"])
def market_basket_analysis():
    if request.method =="POST":
        # Mendapatkan semua file yang diupload
        uploaded_files = request.files.getlist("files")
        # Memastikan setidaknya ada sclatu file yang diupload
        if len(uploaded_files) == 0:
            flash("Harap upload minimal satu file.", 'error')
            return redirect(url_for('market_basket_analysis.market_basket_analysis'))
        
        # Menggabungkan semua file
        df = pd.DataFrame()
        for uploaded_file in uploaded_files:
            if uploaded_file.filename != "":
                try:
                    new_df = pd.read_csv(uploaded_file, sep="\t")
                    df = df.append(new_df)
                except Exception as e:
                    flash(f"Error membaca file: {e}", 'error')
        
        # Check if any data was uploaded after potential errors
        if df.empty:
            flash("No valid data found in uploaded files.", 'error')
            return redirect(url_for('market_basket_analysis.market_basket_analysis'))

        # Perform analysis (assuming you have this defined)
        if df.empty:
            flash("Please upload files or add transactions to begin analysis.", 'info')
        else:
            te_ary = te.fit(df.groupby('Kode Transaksi')['Nama Barang'].apply(
                list)).transform(df.groupby('Kode Transaksi')['Nama Barang'].apply(list))
            df_encoded = pd.DataFrame(te_ary, columns=te.columns_)
            # One-hot encoding using TransactionEncoder
            frequent_itemsets = apriori(df_encoded, min_support=0.1, use_colnames=True)
            rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)
            # ... further rule processing and filtering if needed

            # Sort rules and create unique rules (assuming necessary)
            if rules is not None:
                rules = rules.sort_values(by="lift", ascending=False)
                seen_rules = set()
                unique_rules = []
                for index, row in rules.iterrows():
                    sorted_rule = tuple(sorted(row["antecedents"]) + sorted(row["consequents"]))
                    if sorted_rule not in seen_rules:
                        seen_rules.add(sorted_rule)
                        unique_rules.append(row)
                unique_rules_df = pd.DataFrame(unique_rules)
                unique_rules_html = unique_rules_df.to_html(classes="table table-stripped", index=False)

        return render_template("pages/market_basket_analysis.html", rules=unique_rules_html)

    return render_template("pages/market_basket_analysis.html")