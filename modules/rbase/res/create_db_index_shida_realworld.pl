#!/usr/bin/perl -w

# create index file of the kind
# identifier, image_illum1, image_illum2, image_illum_mixed, ground_truth_image, mcbeth_illum1, mcbeth_illum2, mcbeth_illum_mixed, mask_gt_illum1, mask_gt_illum2, mask_gt_mixed, illum_1, illum_2, illum_mixed
#
# where 'mixed' is an image containing illuminant 1 and illuminant 2

use strict;

my $root_dir = '/net/cv/illum/multi_shida_realworld/';
my $vole = '/disks/data1/riess/code/build/bin/vole';

my %db = ( # identifier first
	'coffee_mug' => {
		image_illum1 => $root_dir . 'img/coffee_mug_l.png',
		image_illum2 => $root_dir . 'img/coffee_mug_r.png',
		image_illum_mixed => $root_dir . 'img/coffee_mug_m.png',
		ground_truth_image => $root_dir . 'gt/coffee_mug.png',
		ground_truth_weights => $root_dir . 'gt/coffee_mug_weights.png',
		mcbeth_illum1 => $root_dir . 'img/coffee_mug_l_gt.png',
		mcbeth_illum2 => $root_dir . 'img/coffee_mug_r_gt.png',
		mcbeth_illum_mixed => $root_dir . 'img/coffee_mug_m_gt.png',
		mask_gt_illum1 => $root_dir . 'gtmask/coffee_mug_l_gt_mask.png',
		mask_gt_illum2 => $root_dir . 'gtmask/coffee_mug_r_gt_mask.png',
		mask_gt_illum_mixed => $root_dir . 'gtmask/coffee_mug_m_gt_mask.png', # image is not really useful
		illum_1 => [0, 0, 0],
		illum_2 => [0, 0, 0],
		illum_mixed => [0, 0, 0], # information is not really useful
	},
	'recipes' => {
		image_illum1 => $root_dir . 'img/recipes_l.png',
		image_illum2 => $root_dir . 'img/recipes_r.png',
		image_illum_mixed => $root_dir . 'img/recipes_m.png',
		ground_truth_image => $root_dir . 'gt/recipes.png',
		ground_truth_weights => $root_dir . 'gt/recipes_weights.png',
		mcbeth_illum1 => $root_dir . 'img/recipes_l_gt.png',
		mcbeth_illum2 => $root_dir . 'img/recipes_r_gt.png',
		mcbeth_illum_mixed => '', # no image available
		mask_gt_illum1 => $root_dir . 'gtmask/recipes_l_gt_mask.png',
		mask_gt_illum2 => $root_dir . 'gtmask/recipes_r_gt_mask.png',
		mask_gt_illum_mixed => '', # no image available
		illum_1 => [0, 0, 0],
		illum_2 => [0, 0, 0],
		illum_mixed => [0, 0, 0], # information is not really useful
	},
	'poster' => {
		image_illum1 => $root_dir . 'img/poster_l.png',
		image_illum2 => $root_dir . 'img/poster_r.png',
		image_illum_mixed => $root_dir . 'img/poster_m.png',
		ground_truth_image => $root_dir . 'gt/poster.png',
		ground_truth_weights => $root_dir . 'gt/poster_weights.png',
		mcbeth_illum1 => $root_dir . 'img/poster_l_gt.png',
		mcbeth_illum2 => $root_dir . 'img/poster_r_gt.png',
		mcbeth_illum_mixed => '', # no image available
		mask_gt_illum1 => $root_dir . 'gtmask/poster_l_gt_mask.png',
		mask_gt_illum2 => $root_dir . 'gtmask/poster_r_gt_mask.png',
		mask_gt_illum_mixed => $root_dir . '', # no image available
		illum_1 => [0, 0, 0],
		illum_2 => [0, 0, 0],
		illum_mixed => [0, 0, 0], # information is not really useful
	}
);

foreach my $id(keys %db) {

	# step 1: extract illuminant color from input image and mask
	my $ill_left  = `$vole getIllumFromMcbeth -V 0 --img.image $db{$id}{mcbeth_illum1} --mask $db{$id}{mask_gt_illum1}`;
	my $ill_right = `$vole getIllumFromMcbeth -V 0 --img.image $db{$id}{mcbeth_illum2} --mask $db{$id}{mask_gt_illum2}`;
	chomp($ill_left);
	chomp($ill_right);

	print "$id illum 1: $ill_left\n";
	print "$id illum 2: $ill_right\n";
	my @tmp1 = split(/\s+/, $ill_left);
	$db{$id}{illum_1} = \@tmp1;

	my @tmp2 = split(/\s+/, $ill_right);
	$db{$id}{illum_2} = \@tmp2;

	# step 2: compute dense illuminant map from two single-illuminant images
	my $i1 = join(',', @{$db{$id}{illum_1}});
	my $i2 = join(',', @{$db{$id}{illum_2}});
	print "$vole gtMultiIllum --img1.image $db{$id}{image_illum1} --img2.image $db{$id}{image_illum2} --illum1 $i1 --illum2 $i2 -O $db{$id}{ground_truth_image}\n";
	print `$vole gtMultiIllum --img1.image $db{$id}{image_illum1} --img2.image $db{$id}{image_illum2} --illum1 $i1 --illum2 $i2 -O $db{$id}{ground_truth_image} -W $db{$id}{ground_truth_weights}`;

}

# step 3: write index file
# identifier, image_illum1, image_illum2, image_illum_mixed, ground_truth_image, mcbeth_illum1, mcbeth_illum2, mcbeth_illum_mixed, mask_gt_illum1, mask_gt_illum2, mask_gt_mixed, illum_1, illum_2, illum_mixed

print "# created from directory " . `pwd` . " using the command $0 " . join(" ", @ARGV) . "\n";
print "# identifier, image_illum1, image_illum2, image_illum_mixed, ground_truth_image, mcbeth_illum1, mcbeth_illum2, mcbeth_illum_mixed, mask_gt_illum1, mask_gt_illum2, mask_gt_mixed, illum_1, illum_2, illum_mixed\n";

foreach my $id(keys %db) {
	print $id.",".
		$db{$id}{image_illum1}.",".$db{$id}{image_illum2}.",".$db{$id}{image_illum_mixed}.",".
		$db{$id}{ground_truth_image}.",".
		$db{$id}{mcbeth_illum1}.",".$db{$id}{mcbeth_illum2}.",".$db{$id}{mcbeth_illum_mixed}.",".
		$db{$id}{mask_gt_illum1}.",".$db{$id}{mask_gt_illum2}.",".$db{$id}{mask_gt_illum_mixed}.",".
		join(",",@{$db{$id}{illum_1}}) . "," . join(",",@{$db{$id}{illum_2}}) . "," . join(",",@{$db{$id}{illum_mixed}})."\n";
}

