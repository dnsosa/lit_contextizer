<?php

include 'db_connect.php';


$sql = "SELECT pair_id,entity1,entity2,notes FROM pairs";
$result = $conn->query($sql);
$data = [];
while ($row = $result->fetch_assoc()) {
	$data[$row['pair_id']] = Array('entity1'=>$row['entity1'], 'entity2'=>$row['entity2'], 'annotations'=>[], 'notes'=>$row['notes']);
}

$sql = "SELECT a.pair_id,o.name FROM annotations a, options o WHERE a.option_id = o.option_id ORDER BY o.option_id";
$result = $conn->query($sql);
while ($row = $result->fetch_assoc()) {
	$data[$row['pair_id']]['annotations'][] = $row['name'];
}

$pair_ids = array_keys($data);
sort($pair_ids);
$pair_count = count($pair_ids);

$sql = "SELECT option_id,name FROM options";
$result = $conn->query($sql);
$options = [];
while ($row = $result->fetch_assoc()) {
	$options[] = $row;
}


$barchart_labels = [];
$barchart_counts = [];
$barchart_colors = [];

$sql = "SELECT o.name,COUNT(*) as count FROM annotations a, options o WHERE a.option_id = o.option_id GROUP BY o.name ORDER BY count DESC";
$result = $conn->query($sql);
while ($row = $result->fetch_assoc()) {	
	$barchart_labels[] = $row['name'];
	$barchart_counts[] = intval($row['count']);
	$barchart_colors[] = "#4682b4";
}

$sql = "SELECT COUNT(DISTINCT(pair_id)) as count FROM annotations";
$result = $conn->query($sql);
if (!$result) {
	$error = mysqli_error($conn);
	echo "<p>ERROR $error with SQL: $sql</p>";
	exit(1);
}
if ($result->num_rows > 0) {
	$row = $result->fetch_assoc();
	$pair_count = $row['count'];
} else {
	$pair_count = 0;
}

if ($pair_count == 1)
	$pair_count_text = "1 pair annotated";
else
	$pair_count_text = "$pair_count pairs annotated"


?>
<!doctype html>
<html lang="en">
	<head>
		<!-- Required meta tags -->
		<meta charset="utf-8" />
		<meta name="viewport" content="width=device-width, initial-scale=1" />

		<!-- Bootstrap CSS -->
		<link href="https://cdn.jsdelivr.net/npm/bootstrap@5.0.0-beta1/dist/css/bootstrap.min.css" rel="stylesheet" integrity="sha384-giJF6kkoqNQ00vy+HMDP7azOuL0xtbfIcaT9wjKHr8RbDVddVHyTfAAsrekwKmP1" crossorigin="anonymous" />

		<style>
			.bd-placeholder-img {
			font-size: 1.125rem;
			text-anchor: middle;
			-webkit-user-select: none;
			-moz-user-select: none;
			user-select: none;
			}

			@media (min-width: 768px) {
			.bd-placeholder-img-lg {
			font-size: 3.5rem;
			}
			}
		</style>

	
		<script src="https://cdn.jsdelivr.net/npm/chart.js@2.9.3/dist/Chart.min.js"></script>

		<title>Contra-annotation</title>
	</head>
	<body>


		<nav class="navbar navbar-expand-md navbar-dark bg-dark mb-4">
			<div class="container-fluid">
				<a class="navbar-brand" href="index.php">Contra-annotation</a>
				<button class="navbar-toggler" type="button" data-bs-toggle="collapse" data-bs-target="#navbarCollapse" aria-controls="navbarCollapse" aria-expanded="false" aria-label="Toggle navigation">
					<span class="navbar-toggler-icon"></span>
				</button>
				<div class="collapse navbar-collapse" id="navbarCollapse">
					<ul class="navbar-nav me-auto mb-2 mb-md-0">
						<li class="nav-item">
							<a class="nav-link active" aria-current="page" href="index.php">View Annotations</a>
						</li>
						<li class="nav-item">
							<a class="nav-link" aria-current="page" href="annotate.php">Annotate</a>
						</li>
					</ul>
				</div>
				<div>
					<ul class="navbar-nav">
						<li class="nav-item" style="width: 200px; text-align: right;">
							<a class="nav-link" aria-current="page" href="index.php"><?php echo $pair_count_text ?></a>
						</li>
					</ul>
				</div>
			</div>
		</nav>

		<main class="container">
	<div class="container">
		<div class="jumbotron">
		
		<h2><?php echo "Overview of $pair_count pairs"; ?></h2>
		
		<div className="chart-container" style="position: relative; height:30vh; width:100%">
			<canvas id="myBarChart"/>
		</div>
	</div>

		<div class="bg-light p-5 rounded">

			<table class="table">
				<thead>
					<tr>
						<th scope="col">#</th>
						<th scope="col">Entity 1</th>
						<th scope="col">Entity 2</th>
						<th scope="col">Annotations</th>
						<th scope="col">Notes</th>
						<th scope="col"></th>
					</tr>
				</thead>
				<tbody>
					<?php
						foreach ($pair_ids as $pair_id) {
							$entity1 = $data[$pair_id]['entity1'];
							$entity2 = $data[$pair_id]['entity2'];
							$notes = $data[$pair_id]['notes'];
							
							$annotation_badges = [];
							foreach ($data[$pair_id]['annotations'] as $anno) {
								$style = 'primary';
								
								if ($anno == 'Skip')
									$style = 'danger';
								elseif ($anno == 'Same Context')
									$style = 'success';
								elseif ($anno == 'Mistake')
									$style = 'info';
								
								$annotation_badges[] = "<span class=\"badge bg-$style\">$anno</span>";
							}
							$annotation_badges_txt = implode(" ",$annotation_badges);
							
							echo "<tr>\n";
							echo "	<th scope='row'>$pair_id</th>\n";
							echo "	<td>$entity1</td>\n";
							echo "	<td>$entity2</td>\n";
							echo "	<td>$annotation_badges_txt</td>\n";
							echo "	<td>$notes</td>\n";
							echo "	<td><a href='annotate.php?pair_id=$pair_id' class='btn btn-dark btn-sm' tabindex='-1' role='button' aria-disabled='true'>Edit</a></td>\n";
							echo "</tr>\n";
						}
					?>
				</tbody>
			</table>
				</div>
				</div>
			</div>


		</main>


		<!-- Optional JavaScript; choose one of the two! -->

		<!-- Option 1: Bootstrap Bundle with Popper -->
		<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.0.0-beta1/dist/js/bootstrap.bundle.min.js" integrity="sha384-ygbV9kiqUc6oa4msXn9868pTtWMgiQaeYH7/t7LECLbyPA2x65Kgf80OJFdroafW" crossorigin="anonymous"></script>

		<!-- Option 2: Separate Popper and Bootstrap JS -->
		<!--
    <script src="https://cdn.jsdelivr.net/npm/@popperjs/core@2.5.4/dist/umd/popper.min.js" integrity="sha384-q2kxQ16AaE6UbzuKqyBE9/u/KzioAlnx2maXQHiDX9d4/zp8Ok3f+M7DPm+Ib6IU" crossorigin="anonymous"></script>
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.0.0-beta1/dist/js/bootstrap.min.js" integrity="sha384-pQQkAEnwaBkjpqZ8RU1fF1AKtTcHJwFl3pblpTlHXybJjHpMYo79HY3hIi4NKxyj" crossorigin="anonymous"></script>
    -->
	
	
	
<script>

var labels = <?php echo json_encode($barchart_labels); ?>;
var counts = <?php echo json_encode($barchart_counts); ?>;
var colors = <?php echo json_encode($barchart_colors); ?>;

// "#3e95cd"

var ctx = document.getElementById("myBarChart");
mychart = new Chart(ctx, {
  type: 'bar',
  data: {
	labels: labels,
	datasets: [ { data: counts, backgroundColor: colors } ]},
  options: {
	legend: { display: false },
	maintainAspectRatio: false,
	scaleShowValues: false,
	scales: {
		xAxes: [{
		ticks: {
			autoSkip: false,
			fontSize: 10
		}
		}],
		yAxes: [{
		  scaleLabel: {
			display: true,
			labelString: '# of pairs'
		  },
		  ticks: {
                beginAtZero: true
            }
		}]
	}
	}
});
</script>
	
	</body>
</html>
