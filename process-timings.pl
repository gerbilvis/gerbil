#!/usr/bin/perl
use feature ":5.10";
no warnings "experimental";
use strict;
#use Term::ANSIColor;

@ARGV == 1 or die "Usage: process-timings.pl <filename>\n";

my $input = $ARGV[0];
#my $name = $ARGV[1];
#my $out = "/tmp/timings/$name";
#system("mkdir -p $out");

open(my $FILE, "<", $input)	or die "Could not open $input $!";

my (%binning, %load, %draw, %mad_avg, %rmse_avg, %mad_bin, %rmse_bin);
print "count\tbinning\tload\tdraw\tmad_avg\trmse_avg\tmad_bin\trmse_bin\n";

my $bin; # declare here so it stays valid
while (<$FILE>) {
    if (/========/ .. /updateBuffer BACK FULL/) {
		my $str = $_;
		my ($number) = $_ =~ /([-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?)/; # match %g
		given ($str) {
			$bin = $number when /binCount/;
			$binning{$bin} = $number when /DistviewBinsTbb/;
			$load{$bin} = $number when /PreparePolylines/;
			$draw{$bin} = $number when /updateBuffer BACK FULL/;
			$mad_avg{$bin} = $number when /NMAD\(avg\)/;
			$rmse_avg{$bin} = $number when /NRMSE\(avg\)/;
			$mad_bin{$bin} = $number when /NMAD\(bin\)/;
			$rmse_bin{$bin} = $number when /NRMSE\(bin\)/;
		}
    }
}

my @keys = sort { $a <=> $b } keys(%binning);
foreach (@keys) {
	my $idx = $_;
	print "$idx\t$binning{$idx}\t$load{$idx}\t$draw{$idx}\t";
	print "$mad_avg{$idx}\t$rmse_avg{$idx}\t$mad_bin{$idx}\t$rmse_bin{$idx}\n";
}
