<?php

include 'db_connect.php';

if (isset($_GET['pair_id'])) {

	$pair_id = $_GET['pair_id'];
	if (!is_numeric($pair_id))
		exit();

} else {
	$sql = "SELECT pair_id FROM pairs WHERE pair_id NOT IN (SELECT pair_id FROM annotations)";
	$result = $conn->query($sql);
	if (!$result) {
		$error = mysqli_error($conn);
		echo "<p>ERROR $error with SQL: $sql</p>";
		exit(1);
	}
	
	if ($result->num_rows > 0) {
		$row = $result->fetch_assoc();
		$pair_id = $row['pair_id'];
	} else {
		$jobdone = true;
	}
}

$sql = "SELECT pair_id, entity1, entity2, a_entity1_text, a_entity2_text, a_simplified_text, a_text, a_year, a_month, a_section_name, a_doc_title, a_journal, a_pmid, a_pmcid, a_doi, b_entity1_text, b_entity2_text, b_simplified_text, b_text, b_year, b_month, b_section_name, b_doc_title, b_journal, b_pmid, b_pmcid, b_doi, notes FROM pairs WHERE pair_id = '$pair_id' LIMIT 1";

if (isset($_GET['savedmsg']))
	$savedmsg = htmlspecialchars($_GET['savedmsg']);
else
	$savedmsg = null;


$result = $conn->query($sql);
if ($result->num_rows > 0) {
	$jobdone = false;
	$data = $result->fetch_assoc();
	
	$pair_id = $data['pair_id'];
	
	$data['moo'] = preg_quote($data['a_entity1_text']);
	
	/*echo "<pre>";
	print_r($data);
	echo "</pre>";*/
	
	$a_text = $data['a_text'];
	if (stripos( $a_text, $data['entity1'] ) !== FALSE)
		$a_text = preg_replace("/\b(".preg_quote($data['entity1'],'/').")\b/i", '<span style="font-weight: bold; color: #3c78d8;">\\1</span>', $a_text);
	else
		$a_text = preg_replace("/\b(".preg_quote($data['a_entity1_text'],'/').")\b/i", '<span style="font-weight: bold; color: #3c78d8;">\\1</span>', $a_text);
		
	if (stripos( $a_text, $data['entity2'] ) !== FALSE)
		$a_text = preg_replace("/\b(".preg_quote($data['entity2'],'/').")\b/i", '<span style="font-weight: bold; color: #674ea7;">\\1</span>', $a_text);
	else
		$a_text = preg_replace("/\b(".preg_quote($data['a_entity2_text'],'/').")\b/i", '<span style="font-weight: bold; color: #674ea7;">\\1</span>', $a_text);
	
	$b_text = $data['b_text'];
	if (stripos( $b_text, $data['entity1'] ) !== FALSE)
		$b_text = preg_replace("/\b(".preg_quote($data['entity1'],'/').")\b/i", '<span style="font-weight: bold; color: #3c78d8;">\\1</span>', $b_text);
	else
		$b_text = preg_replace("/\b(".preg_quote($data['b_entity1_text'],'/').")\b/i", '<span style="font-weight: bold; color: #3c78d8;">\\1</span>', $b_text);
		
	if (stripos( $b_text, $data['entity2'] ) !== FALSE)
		$b_text = preg_replace("/\b(".preg_quote($data['entity2'],'/').")\b/i", '<span style="font-weight: bold; color: #674ea7;">\\1</span>', $b_text);
	else
		$b_text = preg_replace("/\b(".preg_quote($data['b_entity2_text'],'/').")\b/i", '<span style="font-weight: bold; color: #674ea7;">\\1</span>', $b_text);
} else {
	$jobdone = true;
}


$existing_annotations = [];
$sql = "SELECT option_id FROM annotations WHERE pair_id = '$pair_id'";
$result = $conn->query($sql);
while ($row = $result->fetch_assoc()) {	
	$existing_annotations[] = $row['option_id'];
}

$sql = "SELECT option_id,name FROM options";
$result = $conn->query($sql);
if (!$result) {
	$error = mysqli_error($conn);
	echo "<p>ERROR $error with SQL: $sql</p>";
	exit(1);
}
$options = [];
while ($row = $result->fetch_assoc()) {
	$options[] = $row;
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

// 0 pairs annotated

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
							<a class="nav-link" aria-current="page" href="index.php">View Annotations</a>
						</li>
						<li class="nav-item">
							<a class="nav-link active" aria-current="page" href="annotate.php">Annotate</a>
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
			
				<?php if (!is_null($savedmsg)) { ?>
				<div class="alert alert-warning alert-dismissible fade show" role="alert">
					<strong>Saved!</strong> <?php echo $savedmsg ?>
					<button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
				</div>
				<?php } ?>
			
				<div class="row">
					<div class="col">
						<div class="bg-light p-5 rounded" style="height:100%">
							<p class="lead" style="font-family: Arial, Helvetica, sans-serif"><?php echo $a_text ?></p>
							<p>Simplified: <?php echo $data['a_simplified_text'] ?></p>
							<p><span style="font-style: italic;"><?php echo $data['a_doc_title'] ?> (<?php echo $data['a_journal'] ?>)</span><br /><a href="http://doi.org/<?php echo $data['a_doi'] ?>" target="_blank">DOI</a> <a href="https://pubmed.ncbi.nlm.nih.gov/<?php echo $data['a_pmid'] ?>" target="_blank">PubMed</a> <a href="https://www.ncbi.nlm.nih.gov/pmc/articles/PMC<?php echo $data['a_pmcid'] ?>/" target="_blank">PMC</a></p>
						</div>
					</div>
					<div class="col">
						<div class="bg-light p-5 rounded" style="height:100%">
							<p class="lead" style="font-family: Arial, Helvetica, sans-serif"><?php echo $b_text ?></p>
							<p>Simplified: <?php echo $data['b_simplified_text'] ?></p>
							<p><span style="font-style: italic;"><?php echo $data['b_doc_title'] ?> (<?php echo $data['b_journal'] ?>)</span><br /><a href="http://doi.org/<?php echo $data['b_doi'] ?>" target="_blank">DOI</a> <a href="https://pubmed.ncbi.nlm.nih.gov/<?php echo $data['b_pmid'] ?>" target="_blank">PubMed</a> <a href="https://www.ncbi.nlm.nih.gov/pmc/articles/PMC<?php echo $data['b_pmcid'] ?>/" target="_blank">PMC</a></p>
						</div>
					</div>
				</div>
			</div>

			<!-- <p class="lead" style="font-family: Arial, Helvetica, sans-serif"><span style="font-weight: bold; color: #3c78d8;">Mec1/ATR</span> inhibits mitotic <span style="font-weight: bold; color: #674ea7;">Cdk1</span> activity through downstream effector kinases, Rad53/Chk2 and Swe1/Wee1.</p> -->


			<form action="add_annotation.php" method="post">
				<input type="hidden" name="pair_id" value="<?php echo $pair_id; ?>" />
				<input type="hidden" name="entity1" value="<?php echo $data['entity1']; ?>" />
				<input type="hidden" name="entity2" value="<?php echo $data['entity2']; ?>" />
				<div class="row" style="margin-top:20px">
					<div class="col-sm-7" style="text-align: center;">
							<?php
								foreach ($options as $option) {
									$option_id = $option['option_id'];
									$name = $option['name'];
									
									$style = 'primary';
									
									if ($name == 'Skip')
										$style = 'danger';
									elseif ($name == 'Same Context')
										$style = 'success';
									elseif ($name == 'Mistake')
										$style = 'info';
									
									$is_checked = '';			
									if (in_array($option_id, $existing_annotations)) {
										$is_checked = 'checked';
									}
									
									echo "<input name='option_$option_id' type='checkbox' class='btn-check' id='btn-check-$option_id' autocomplete='off' $is_checked/>\n";
									echo "<label class='btn btn-outline-$style' style='margin-bottom:4px' for='btn-check-$option_id'>$name</label>\n";
								}
							?>

							<input name="new_option" class="form-control" style="display:inline; width:30%; vertical-align: middle; margin-bottom:4px" type="text" placeholder="Add new annotation" aria-label="Space for new annotation" />
					</div>
					<div class="col-sm-3">
							<div class="mb-2">
								<textarea name="notes" class="form-control" id="annotation_notes" rows="2" placeholder="Notes" ><?php echo $data['notes'] ?></textarea>
							</div>
						


					</div>
						<!-- <button type="submit" class="btn btn-warning">Prev</button> -->
						<!-- <button type="submit" class="btn btn-warning">Next</button> -->
					<div class="col-sm-1">
						<button type="submit" class="btn btn-primary">Save &amp; Next</button>
					</div>
					<div class="col-sm-1">
					</div>
				</div>
				<div style="text-align: center;">
				</div>
			</form>


		</main>


		<!-- Optional JavaScript; choose one of the two! -->

		<!-- Option 1: Bootstrap Bundle with Popper -->
		<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.0.0-beta1/dist/js/bootstrap.bundle.min.js" integrity="sha384-ygbV9kiqUc6oa4msXn9868pTtWMgiQaeYH7/t7LECLbyPA2x65Kgf80OJFdroafW" crossorigin="anonymous"></script>

		<!-- Option 2: Separate Popper and Bootstrap JS -->
		<!--
    <script src="https://cdn.jsdelivr.net/npm/@popperjs/core@2.5.4/dist/umd/popper.min.js" integrity="sha384-q2kxQ16AaE6UbzuKqyBE9/u/KzioAlnx2maXQHiDX9d4/zp8Ok3f+M7DPm+Ib6IU" crossorigin="anonymous"></script>
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.0.0-beta1/dist/js/bootstrap.min.js" integrity="sha384-pQQkAEnwaBkjpqZ8RU1fF1AKtTcHJwFl3pblpTlHXybJjHpMYo79HY3hIi4NKxyj" crossorigin="anonymous"></script>
    -->
	</body>
</html>
