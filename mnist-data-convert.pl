#!/usr/bin/perl

use strict;
use Data::Dumper;

# This script expects MNIST data in CSV format
# as downloaded from https://www.kaggle.com/c/digit-recognizer/data.
# It outputs three datasets in Lua format:
# - train - training data containing all except $NewClasses classes/labels
# - test - test data containing the same classes and $testratio of the available data
# - new - the remaining $NewClasses classes for evaluating the model on unseen classes

my $infilename = 'data/mnist/train.csv'; # input file, the MNIST data in CSV format
my $TestRatio = 0.3; # The ratio of items to add to the test set
my $Classes = [0,1,2,4,5,6,8,9,3,7]; # The MNIST classes reordered so that new classes would be at the end
my $NewClasses = 2; # Number of classes at the end of $Classes to include in the new set
my $NewRatio = 0.1; # Only add this ratio of items to the "new" set
my $NewFromTestRatio = 0.01; # Add this many elements from the test set to the "new" set
my $TrainFile = 'data/train.lua'; # Output file
my $TestFile = 'data/test.lua'; # Output file
my $NewFile = 'data/new.lua'; # Output file

my @trainset;
my @testset;
my @newset;

# Used to reorder classes/labels
my %Classmap;
for (my $i=0; $i<@$Classes; $i++) {
   $Classmap{$Classes->[$i]} = $i;
}

# writefile($filename, $contents)
sub writefile {
  my $filename = shift;
  my $contents = shift;
  
  open(my $fh, '>', $filename) or die "Cannot open $filename: $!";
  print $fh $contents;
  close($fh);
}

# Write Lua code that initialises a Torch tensor with the dataset
sub write_lua {
   my $data = shift; # [ [LABEL,[PIXELS]], ... ]
   my $filename = shift; # output file
   my $is_new = shift; # bool; whether we are writing the new dataset
   
   my $numclass = $is_new ? ($NewFromTestRatio == 0 ? $NewClasses : 10) : (10 - $NewClasses);
   my $datasize = scalar(@$data);
   my $numpixel = scalar(@{$data->[0]->[1]});

   my $o = <<THEEND;
require 'torch'
local x = torch.Tensor($datasize, $numpixel):fill(0)
local targetlabels = torch.Tensor($datasize, 1)
THEEND

   # If $is_new, y is unused
   $o .= $is_new ? "local y = torch.Tensor(1)\n" : "local y = torch.Tensor($datasize, $numclass):fill(-1)\n";

   for(my $i=0; $i<$datasize; $i++) {
      my $ii = $i+1;
      $o .= "function __mnist${ii}()\n";
      for(my $j=0; $j<$numpixel; $j++) {
         $o .= 'x['.$ii.']['.($j+1).']='.$data->[$i]->[1]->[$j]."\n" if $data->[$i]->[1]->[$j];
      }
      $o .= 'y['.$ii.']['.($data->[$i]->[0]+1)."]=1\n" unless $is_new;
      $o .= 'targetlabels['.$ii.'][1] = '.($data->[$i]->[0]+1)."\n";
      $o .= "end\n__mnist${ii}()\n";
   }
   
   $o .= "return {numclusters=$numclass,x=x,y=y,targetlabels=targetlabels}\n";

   writefile($filename, $o);
}

# Main code

my $header = 1;
open(my $fh, '<', $infilename) or die "Cannot open $infilename: $!";
while(<$fh>) {
   if($header) { $header = 0; next; }
   s/\s+|\s+$//g;
   my @line = split(/,/, $_);
   my $label = $Classmap{shift(@line)};
   
   my $outline = [$label, \@line];
   if ($label >= 10 - $NewClasses) {
      if (rand() < $NewRatio) {
         push @newset, $outline;
      }
   }
   elsif (rand() < $TestRatio) {
      if (rand() < $NewFromTestRatio) {
         push @newset, $outline;
      }
      push @testset, $outline;
   }
   else {
      push @trainset, $outline;
   }
}
close($fh);

print "Train set size " . scalar(@trainset) . "\n";
print "Test set size " . scalar(@testset) . "\n";
print "New set size " . scalar(@newset) . "\n";

write_lua(\@trainset, $TrainFile, 0);
write_lua(\@testset, $TestFile, 0);
write_lua(\@newset, $NewFile, 1);
