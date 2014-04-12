#!/usr/bin/perl
use feature ":5.10";
no warnings "experimental";
use strict;
#use Term::ANSIColor;

@ARGV == 2 or die "Usage: process-table.pl <filename> <dataset>\n";

my %name = (
	'dcmall'	=> 'D.C. Mall',
	'cave'		=> 'CAVE',
	'foster'	=> 'Foster',
);

my %dim = (
	'dcmall'	=> 191,
	'cave'		=> 31,
	'foster'	=> 33,
);

my @bins = ( 32, 64, 256 );

my $input = $ARGV[0];
my $dataset = $ARGV[1];

open(my $FILE, "<", $input)	or die "Could not open $input $!";

print "count\tbinning+load\tdraw\trmse_avg\tmad_avg\n";

my $cur = 0;
my $curbin = $bins[$cur];
while (<$FILE>) {
    if (/^$curbin/) {
		my @stuff = split(/\s+/);
		my $prep = $stuff[1] + $stuff[2];
		print $name{$dataset} unless $cur > 0;
		print " & \$$stuff[0]^{$dim{$dataset}}\$";
		printf(" & \$%.3f\$ & \$%.3f\$", $prep, $stuff[3]);
		printf(" & \$%.3f\$ & \$%.3f\$ \\\\\n", $stuff[5], $stuff[4]);
		
		$curbin = $bins[++$cur];
    }
}
