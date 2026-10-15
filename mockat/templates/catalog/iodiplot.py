<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Plans [{{obs.id}}]</title>
    <link rel="stylesheet" href="https://pyscript.net/alpha/pyscript.css" />
    <script defer src="https://pyscript.net/alpha/pyscript.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <link rel="stylesheet" href="{{staturl}}css/theme.css" />

</head>
<body>
    <script>
    const chart = new Chart("chart", {
	  type: "line",
	  data: {
		datasets: [{
			borderColor: "#007cfb",
			backgroundColor: "#0062c5",
		  }
		]
	  },
	  options: {
		plugins: {
		  legend: {display: false},
		  tooltip: {enabled: false},
		},
		scales: {
		  x: {display: true},
		  y: {display: false}
		},
	  }
	})

	function updateChart(xValues, yValues) {
	  chart.data.labels = xValues
	  chart.data.datasets[0].data = yValues
	  chart.update()
	}
	</script>
  <fieldset>
    <legend>Contrast<sub>sep</sub></legend>
    <input id="range1"
      type="range" min="1" max="10" step="0.1" value="5">
    <label for="range1">1</label>
  </fieldset>
  <fieldset>
    <legend>Length</legend>
    <input id="range2"
      type="range" min="0" max="1000" step="1" value="10">
    <label for="range2">400</label> sec
  </fieldset>
  <py-button label="Click me" styles="btn big">
	def on_click(event):
		print(event)
	</py-button>
  <h1>{{obs.desc}}</h1>
  <div id="topten" style="font-size:150%;color:#800000"></div>
  <canvas id="chart"></canvas>
  <py-script>
	  from pyodide.http import open_url
	  from pyodide import to_js
	  data=open_url("{% url 'data' obs.pk %}{{query}}").read()
	  lines=data.split('\n')[:100]
	  from js import updateChart
	  specs=[a.split()[:3] for a in lines]
	  times=[float(a[0]) for a in specs]
	  vals=[float(a[1]) for a in specs]
	  cont='<br/> '.join([str(s) for s in specs[:3]])
	  cont='<br/> '.join([str(s) for s in vals[:30]])
	  Element('topten').write(cont)
	  updateChart((to_js(times), to_js(vals)))
  </py-script>
</body>
</html>
