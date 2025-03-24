from flask import Flask, render_template, request
import numpy as np
import matplotlib
matplotlib.use('Agg')  # <-- use a non-GUI backend that works with Flask
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import io
import base64

app = Flask(__name__)

# --- Model Functions ---
def linear_model(Q, m, b):
    return m * Q + b

def quadratic_model(Q, a, b, c):
    return a * Q**2 + b * Q + c

def nonlinear_price_model(Q, a, b):
    return a * Q**b

def find_break_even_points(Q_range, TR, TC):
    diff = TR - TC
    sign_changes = np.where(np.diff(np.sign(diff)))[0]
    return Q_range[sign_changes]

# --- Main Route ---
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            # --- Parse Input ---
            cost_data = request.form['cost_data'].strip().split('\n')
            Q_cost, TC_data = zip(*[map(float, line.split(",")) for line in cost_data])
            Q_cost = np.array(Q_cost)
            TC_data = np.array(TC_data)

            revenue_data = request.form['revenue_data'].strip().split('\n')
            Q_rev, P_rev = zip(*[map(float, line.split(",")) for line in revenue_data])
            Q_rev = np.array(Q_rev)
            P_rev = np.array(P_rev)

            q_min = int(request.form['qmin'])
            q_max = int(request.form['qmax'])

            # --- Fit Models ---
            popt_lin, _ = curve_fit(linear_model, Q_cost, TC_data)
            popt_quad, _ = curve_fit(quadratic_model, Q_cost, TC_data)
            popt_price, _ = curve_fit(nonlinear_price_model, Q_rev, P_rev)

            # --- Simulation ---
            Q_full = np.linspace(1, 2000, 2000)
            TR_full = nonlinear_price_model(Q_full, *popt_price) * Q_full
            TC_full = quadratic_model(Q_full, *popt_quad)
            break_even_points = find_break_even_points(Q_full, TR_full, TC_full)

            Q_sim = np.linspace(0, q_max, 400)
            Q_sim_safe = Q_sim.copy()
            Q_sim_safe[Q_sim_safe == 0] = 1e-6
            P_sim = nonlinear_price_model(Q_sim_safe, *popt_price)
            TR = P_sim * Q_sim_safe
            TC_sim = quadratic_model(Q_sim, *popt_quad)
            profit = TR - TC_sim

            max_profit = np.max(profit)
            Q_max_profit = Q_sim[np.argmax(profit)]

            # --- Plotting ---
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

            ax1.scatter(Q_cost, TC_data, color='black', label="User Data")
            ax1.plot(Q_sim, linear_model(Q_sim, *popt_lin), 'r-', label=f"Linear Fit")
            ax1.plot(Q_sim, quadratic_model(Q_sim, *popt_quad), 'b--', label=f"Quadratic Fit")
            ax1.set_title("Cost Function")
            ax1.set_xlabel("Quantity")
            ax1.set_ylabel("Total Cost")
            ax1.legend()
            ax1.grid(True)

            ax2.plot(Q_sim, profit, 'purple', linestyle='--', label="Profit")
            ax2.plot(Q_max_profit, max_profit, 'o', color='purple', label=f"Max Profit: ${max_profit:.2f} at Q={Q_max_profit:.0f}")
            for q in break_even_points:
                ax2.axvline(q, color='gray', linestyle='dotted', label=f"Break-Even: {q:.0f}")
            ax2.set_title("Profit Function")
            ax2.set_xlabel("Quantity")
            ax2.set_ylabel("Profit")
            ax2.legend()
            ax2.grid(True)

            ax3.plot(Q_sim, TR, 'blue', label="Total Revenue")
            ax3.plot(Q_sim, TC_sim, 'red', label="Total Cost")
            ax3.fill_between(Q_sim, TC_sim, TR, where=TR>=TC_sim, color='green', alpha=0.2, label='Profit')
            ax3.fill_between(Q_sim, TC_sim, TR, where=TR<TC_sim, color='red', alpha=0.1, label='Loss')
            for q in break_even_points:
                ax3.axvline(q, color='gray', linestyle='dotted')
            ax3.set_title("Revenue vs Cost")
            ax3.set_xlabel("Quantity")
            ax3.set_ylabel("Dollars")
            ax3.legend()
            ax3.grid(True)

            plt.tight_layout()
            img = io.BytesIO()
            plt.savefig(img, format='png')
            img.seek(0)
            plot_url = base64.b64encode(img.getvalue()).decode()
            plt.close()

            return render_template("index.html",
                plot_url=plot_url,
                max_profit=round(max_profit, 2),
                Q_max_profit=round(Q_max_profit, 1),
                break_even_points=[(round(q, 2), round(nonlinear_price_model(q, *popt_price), 2)) for q in break_even_points],
                cost_data=request.form['cost_data'],
                revenue_data=request.form['revenue_data'],
                qmin=q_min,
                qmax=q_max
            )



        except Exception as e:
            return f"<h3>Error:</h3><pre>{e}</pre>"

    return render_template("index.html")

if __name__ == '__main__':
    app.run(debug=True)